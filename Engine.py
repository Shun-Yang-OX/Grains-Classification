# Engine.py

import torch
import os
import torch.distributed as dist
from torch import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F
import utils_setup
from collections import deque

def setup_ddp(rank, world_size):
    """
    Initialize the Distributed Data Parallel (DDP) environment.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes (typically equal to the number of GPUs).
    """
    # Set environment variables for master address and port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the default process group for distributed communication
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    # Set the current CUDA device to the given rank
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """
    Clean up the distributed process group.
    """
    dist.destroy_process_group()

def train_and_validate_one_epoch_ddp(model, data_loader_train, data_loader_val, train_sampler, optimizer, scaler, device, epoch, rank, scheduler, experiment=None, global_step=0):
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
        experiment: Comet.ml experiment object for logging.
        global_step (int): Global step counter.
        
    Returns:
        average_train_loss (float): Average training loss for the epoch.
        average_val_loss (float): Average validation loss for the epoch.
        global_step (int): Updated global step counter after epoch.
    """
    model.train()  # Switch model to training mode
    # Set the epoch for the sampler to ensure proper shuffling in distributed training
    train_sampler.set_epoch(epoch)
    
    total_steps_per_epoch = len(data_loader_train)
    total_train_loss = 0.0

    # Create a tqdm progress bar for training, only visible on rank 0
    if rank == 0:
        train_progress_bar = tqdm(data_loader_train, desc=f"Rank {rank} | Epoch {epoch} [Training]", position=rank, leave=True)
    else:
        train_progress_bar = data_loader_train

    # -------------------- Training Loop --------------------
    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        # Move data to the correct device
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Reset gradients

        # Mixed precision training context
        with autocast("cuda"):
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

        # Scale the loss and perform backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()  # Accumulate loss

        # Step the learning rate scheduler if provided
        if scheduler:
            scheduler.step()

        # Update progress bar with current loss (only for main process)
        if rank == 0:
            train_progress_bar.set_postfix(loss=loss.item())

        # Update global_step on rank 0 and broadcast its value to all processes
        if rank == 0:
            global_step += 1
        global_step_tensor = torch.tensor(global_step, dtype=torch.int64, device=device)
        dist.broadcast(global_step_tensor, src=0)
        global_step = global_step_tensor.item()

        # Every 20 steps, log metrics to Comet.ml (only on rank 0)
        if rank == 0 and experiment is not None and (global_step % 20 == 0):
            current_loss = loss.item()
            learning_rate = optimizer.param_groups[0]['lr']

            # Log training loss and learning rate
            utils_setup.log_metrics_to_comet(
                experiment,
                global_step,
                train_loss=current_loss,
                learning_rate=learning_rate,
                duration=None,
                rank=rank
            )

            # Calculate output probabilities for positive and negative classes
            probs = F.softmax(outputs, dim=1)
            pos_probs = probs[:, 1].detach().cpu().numpy()
            neg_probs = probs[:, 0].detach().cpu().numpy()

            # Log histograms of output probabilities to Comet.ml
            experiment.log_histogram_3d(
                pos_probs,
                name='Train/Output_Probabilities_PosClass',
                step=global_step
            )
            experiment.log_histogram_3d(
                neg_probs,
                name='Train/Output_Probabilities_NegClass',
                step=global_step
            )

            # Log histograms of logits for each class
            logits = outputs.detach().cpu().numpy()
            experiment.log_histogram_3d(
                logits[:, 0],
                name='Train/Logits_Class0',
                step=global_step
            )
            experiment.log_histogram_3d(
                logits[:, 1],
                name='Train/Logits_Class1',
                step=global_step
            )

            # Log histograms of predicted labels
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            experiment.log_histogram_3d(
                preds,
                name='Train/Predicted_Labels',
                step=global_step
            )

    # Aggregate training loss across all processes
    total_train_loss_tensor = torch.tensor(total_train_loss, dtype=torch.float32, device=device)
    dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)
    total_train_loss_tensor = total_train_loss_tensor / dist.get_world_size()
    # Compute average training loss
    average_train_loss = total_train_loss_tensor.item() / len(data_loader_train)

    if rank == 0:
        print(f"Rank {rank} | Epoch [{epoch}] Training Completed | Average Train Loss (all GPUs): {average_train_loss:.4f}")

    # -------------------- Validation Loop --------------------
    model.eval()  # Switch model to evaluation mode
    total_val_loss = 0.0

    # Create a tqdm progress bar for validation, only visible on rank 0
    if rank == 0:
        val_progress_bar = tqdm(data_loader_val, desc=f"Rank {rank} | Epoch {epoch} [Validation]", position=rank, leave=True)
    else:
        val_progress_bar = data_loader_val

    all_outputs = []  # To store model outputs for all validation samples
    all_labels = []   # To store corresponding labels

    # Disable gradient computation during validation
    with torch.no_grad():
        for images, labels in val_progress_bar:
            images, labels = images.to(device), labels.to(device)

            with autocast("cuda"):
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)

            total_val_loss += loss.item()
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

            if rank == 0:
                val_progress_bar.set_postfix(loss=loss.item())

    # Aggregate validation loss across all processes
    total_val_loss_tensor = torch.tensor(total_val_loss, dtype=torch.float32, device=device)
    dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
    total_val_loss_tensor = total_val_loss_tensor / dist.get_world_size()
    # Compute average validation loss
    average_val_loss = total_val_loss_tensor.item() / len(data_loader_val)

    # After validation, log metrics and histograms on rank 0
    if rank == 0 and experiment is not None:
        utils_setup.log_metrics_to_comet(
            experiment,
            global_step,
            validation_loss=average_val_loss,
            learning_rate=None,
            duration=None,
            rank=rank
        )

        # Concatenate outputs from all validation batches
        all_outputs = torch.cat(all_outputs, dim=0)
        # Calculate probabilities for each class
        probs = F.softmax(all_outputs, dim=1)
        pos_probs = probs[:, 1].numpy()
        neg_probs = probs[:, 0].numpy()

        # Log histograms of output probabilities for validation set
        experiment.log_histogram_3d(
            pos_probs,
            name='Validation/Output_Probabilities_PosClass',
            step=global_step
        )
        experiment.log_histogram_3d(
            neg_probs,
            name='Validation/Output_Probabilities_NegClass',
            step=global_step
        )

        # Log histograms of logits for each class on validation set
        logits = all_outputs.numpy()
        experiment.log_histogram_3d(
            logits[:, 0],
            name='Validation/Logits_Class0',
            step=global_step
        )
        experiment.log_histogram_3d(
            logits[:, 1],
            name='Validation/Logits_Class1',
            step=global_step
        )

        # Log histograms of predicted labels on validation set
        preds = torch.argmax(all_outputs, dim=1).numpy()
        experiment.log_histogram_3d(
            preds,
            name='Validation/Predicted_Labels',
            step=global_step
        )

        print(f"Rank {rank} | Epoch [{epoch}] Validation Completed | Average Val Loss (all GPUs): {average_val_loss:.4f}")

    # Return average training/validation loss and updated global step
    return average_train_loss, average_val_loss, global_step
