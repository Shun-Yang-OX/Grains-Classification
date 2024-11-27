import torch
import os
import torch.distributed as dist
from torch.amp import autocast
from tqdm import tqdm
import torch.nn.functional as F
import utils

def setup_ddp(rank, world_size):
    """
    Set up Distributed Data Parallel (DDP) environment.
    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes (GPUs).
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up the DDP environment by destroying the process group."""
    dist.destroy_process_group()

def train_and_validate_one_epoch_ddp(model, data_loader_train, data_loader_val, train_sampler, optimizer, scaler, device, epoch, rank, scheduler, tb_writer=None):
    """
    Train and validate the model for one epoch using DDP for image classification with ResNet.
    Args:
        model (torch.nn.Module): The model to train and validate.
        data_loader_train (DataLoader): DataLoader for the training set.
        data_loader_val (DataLoader): DataLoader for the validation set.
        train_sampler (Sampler): DistributedSampler for training data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        device (torch.device): Device for training (GPU).
        epoch (int): Current epoch number.
        rank (int): Rank of the current process.
    Returns:
        tuple: Average training loss and validation loss.
    """
    model.train()  # Switch to training mode
    train_sampler.set_epoch(epoch)
    
    total_train_loss = 0.0
    total_steps_per_epoch = len(data_loader_train)  # 获取每个 epoch 的总步数
    num_logs_per_epoch = 30  # 您希望每个 epoch 记录的点数
    log_interval = max(1, total_steps_per_epoch // num_logs_per_epoch)  # 计算日志记录间隔
    global_step = epoch * total_steps_per_epoch  # 计算当前 epoch 开始时的全局 step 数

    # Create tqdm progress bar for the training set
    if rank == 0:
        train_progress_bar = tqdm(data_loader_train, desc=f"Rank {rank} | Epoch {epoch} [Training]", position=rank, leave=True)
    else:
        train_progress_bar = data_loader_train  # Other ranks use a normal iterator

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)  # Cross-entropy loss for classification

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_train_loss += loss.item()

        if scheduler:
            scheduler.step() 
        # Update the description of the tqdm progress bar
        if rank == 0:
            train_progress_bar.set_postfix(loss=loss.item())
        
        # update  global_step
        global_step += 1

        # 仅在主进程上记录
        if rank == 0 and (batch_idx + 1) % log_interval == 0:
            # 记录训练损失到 TensorBoard
            learning_rate = optimizer.param_groups[0]['lr']
            utils.log_metrics_to_tensorboard(tb_writer, global_step, train_loss=loss.item(), learning_rate=learning_rate, rank=rank)

    # Use all_reduce to calculate the total training loss across all GPUs
    total_train_loss_tensor = torch.tensor(total_train_loss, dtype=torch.float32, device=device)
    dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)  # Aggregate loss across all GPUs
    total_train_loss_tensor = total_train_loss_tensor / dist.get_world_size()  # Calculate average loss

    average_train_loss = total_train_loss_tensor.item() / len(data_loader_train)

    if rank == 0:
        print(f"Rank {rank} | Epoch [{epoch}] Training Completed | Average Train Loss (all GPUs): {average_train_loss:.4f}")

    # Start the validation phase
    total_val_loss = 0.0

    # Create tqdm progress bar for the validation set
    if rank == 0:
        val_progress_bar = tqdm(data_loader_val, desc=f"Rank {rank} | Epoch {epoch} [Validation]", position=rank, leave=True)
    else:
        val_progress_bar = data_loader_val

    model.eval()  # Switch to evaluation mode
    with torch.no_grad():  # No gradient calculation during validation
        for images, labels in val_progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Mixed precision inference
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)

            total_val_loss += loss.item()

            if rank == 0:
                val_progress_bar.set_postfix(loss=loss.item())

    # Use all_reduce to calculate the total validation loss across all GPUs
    total_val_loss_tensor = torch.tensor(total_val_loss, dtype=torch.float32, device=device)
    dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)  # Aggregate loss across all GPUs
    total_val_loss_tensor = total_val_loss_tensor / dist.get_world_size()  # Calculate average loss

    average_val_loss = total_val_loss_tensor.item() / len(data_loader_val)

    # 在主进程上记录验证损失
    if rank == 0:
        utils.log_metrics_to_tensorboard(tb_writer, global_step, validation_loss=average_val_loss, rank=rank)


    if rank == 0:
        print(f"Rank {rank} | Epoch [{epoch}] Validation Completed | Average Val Loss (all GPUs): {average_val_loss:.4f}")

    return average_train_loss, average_val_loss
