import os
import torch
import torch.distributed as dist
from torch import autocast, GradScaler
import torch.nn.functional as F
from tqdm import tqdm
from collections import deque

import utils_setup
import wandb

def setup_ddp(rank, world_size):
    """
    Initialize the Distributed Data Parallel (DDP) environment.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes (typically equal to the number of GPUs).
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """
    Clean up the distributed process group.
    """
    dist.destroy_process_group()


def train_and_validate_one_epoch_ddp(
    model,
    data_loader_train,
    data_loader_val,
    train_sampler,
    optimizer,
    scaler,
    device,
    epoch,
    rank,
    scheduler,
    experiment=None,
    global_step=0
):
    """
    Train and validate the model for one epoch using Distributed Data Parallel (DDP).

    Args:
        model: The model to train.
        data_loader_train: DataLoader for training data.
        data_loader_val: DataLoader for validation data.
        train_sampler: Sampler used to partition data among processes.
        optimizer: Optimizer for updating model weights.
        scaler: GradScaler for mixed precision training.
        device: The device on which computation is performed.
        epoch (int): Current epoch number.
        rank (int): Rank of the current process.
        scheduler: Learning rate scheduler.
        experiment: wandb Run object for logging.
        global_step (int): Global step counter.

    Returns:
        average_train_loss (float): Average training loss for the epoch.
        average_val_loss (float): Average validation loss for the epoch.
        global_step (int): Updated global step counter after epoch.
    """
    # -------------------- TRAINING --------------------
    model.train()
    train_sampler.set_epoch(epoch)  # Ensure proper shuffling in distributed mode

    total_steps_per_epoch = len(data_loader_train)
    total_train_loss = 0.0

    # tqdm progress bar (only rank=0 prints it)
    if rank == 0:
        train_progress_bar = tqdm(
            data_loader_train,
            desc=f"Rank {rank} | Epoch {epoch} [Training]",
            position=rank,
            leave=True
        )
    else:
        train_progress_bar = data_loader_train

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()

        # Learning rate scheduling (step-level)
        if scheduler:
            scheduler.step()

        # Update progress bar
        if rank == 0:
            train_progress_bar.set_postfix(loss=loss.item())

        # Synchronize global_step to all processes
        if rank == 0:
            global_step += 1
        global_step_tensor = torch.tensor(global_step, dtype=torch.int64, device=device)
        dist.broadcast(global_step_tensor, src=0)
        global_step = global_step_tensor.item()

        # Every few steps, log to wandb (handled only in rank=0)
        if rank == 0 and experiment is not None and (global_step % 200 == 0):
            current_loss = loss.item()
            learning_rate = optimizer.param_groups[0]['lr']

            # Log training loss and learning rate
            utils_setup.log_metrics_to_wandb(
                experiment,
                global_step,
                train_loss=current_loss,
                learning_rate=learning_rate,
                duration=None,
                rank=rank
            )

            # Log histograms for output probabilities, logits, and predicted labels
            probs = F.softmax(outputs, dim=1)  # shape: (batch, num_classes)
            probs_np = probs.detach().cpu().numpy()
            logits_np = outputs.detach().cpu().numpy()

            for class_idx in range(probs_np.shape[1]):
                experiment.log({
                    f"Train/Output_Probabilities_Class{class_idx}": wandb.Histogram(probs_np[:, class_idx])
                }, step=global_step)
                experiment.log({
                    f"Train/Logits_Class{class_idx}": wandb.Histogram(logits_np[:, class_idx])
                }, step=global_step)

            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            experiment.log({
                'Train/Predicted_Labels': wandb.Histogram(preds)
            }, step=global_step)

    # Aggregate training loss across GPUs
    total_train_loss_tensor = torch.tensor(total_train_loss, dtype=torch.float32, device=device)
    dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)
    total_train_loss_tensor = total_train_loss_tensor / dist.get_world_size()

    # Calculate average training loss (sum of batch losses divided by number of batches)
    average_train_loss = total_train_loss_tensor.item() / len(data_loader_train)

    if rank == 0:
        print(f"Rank {rank} | Epoch [{epoch}] Training Completed | Avg Train Loss (all GPUs): {average_train_loss:.4f}")

    # -------------------- VALIDATION --------------------
    model.eval()
    total_val_loss = 0.0

    if rank == 0:
        val_progress_bar = tqdm(
            data_loader_val,
            desc=f"Rank {rank} | Epoch {epoch} [Validation]",
            position=rank,
            leave=True
        )
    else:
        val_progress_bar = data_loader_val

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Check if in multi-crop mode (B, num_crops, C, H, W)
            if images.dim() == 5:
                B, num_crops, C, H, W = images.shape
                # Flatten multi-crop data to (B*num_crops, C, H, W)
                images = images.view(B * num_crops, C, H, W)
                with autocast("cuda"):
                    outputs = model(images)  # Output shape: (B*num_crops, num_classes)
                # Reshape output to (B, num_crops, num_classes) and average over crops
                outputs = outputs.view(B, num_crops, -1).mean(dim=1)
            else:
                with autocast("cuda"):
                    outputs = model(images)
            
            loss = F.cross_entropy(outputs, labels)
            total_val_loss += loss.item()

            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

            if rank == 0:
                val_progress_bar.set_postfix(loss=loss.item())

    # Aggregate validation loss across GPUs
    total_val_loss_tensor = torch.tensor(total_val_loss, dtype=torch.float32, device=device)
    dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
    total_val_loss_tensor = total_val_loss_tensor / dist.get_world_size()

    average_val_loss = total_val_loss_tensor.item() / len(data_loader_val)

    # Only rank==0 performs wandb logging and printing
    if rank == 0 and experiment is not None:
        utils_setup.log_metrics_to_wandb(
            experiment,
            global_step,
            validation_loss=average_val_loss,
            rank=rank
        )

        # Concatenate outputs from all batches and compute histograms
        all_outputs = torch.cat(all_outputs, dim=0)  # shape: (total_val_samples, num_classes)
        logits_np = all_outputs.numpy()
        probs = F.softmax(all_outputs, dim=1).numpy()

        for class_idx in range(probs.shape[1]):
            experiment.log({
                f'Validation/Output_Probabilities_Class{class_idx}': wandb.Histogram(probs[:, class_idx])
            }, step=global_step)
            experiment.log({
                f'Validation/Logits_Class{class_idx}': wandb.Histogram(logits_np[:, class_idx])
            }, step=global_step)

        preds = torch.argmax(all_outputs, dim=1).numpy()
        experiment.log({
            'Validation/Predicted_Labels': wandb.Histogram(preds)
        }, step=global_step)

        print(f"Rank {rank} | Epoch [{epoch}] Validation Completed | Avg Val Loss (all GPUs): {average_val_loss:.4f}")

    return average_train_loss, average_val_loss, global_step
