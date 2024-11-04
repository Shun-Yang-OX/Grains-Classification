from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
import logging
import random
import numpy as np
import torch
import re
import os

## Logger ##
######################################################################################################
def setup_logging(log_dir, rank):
    # Set up logging if on the main process (rank 0)
    if rank == 0:
        log_file = os.path.join(log_dir, f'train_rank_{rank}.log')
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        tb_writer = SummaryWriter(log_dir=log_dir)
    else:
        tb_writer = None
    return tb_writer

def log_metrics_to_file(epoch, train_loss, validation_loss, learning_rate, duration, rank):
    # Log metrics to a file if on the main process (rank 0)
    if rank == 0:
        logging.info(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, LR: {learning_rate:.6f}, Duration: {duration:.2f}s")

def log_metrics_to_tensorboard(tb_writer, epoch, train_loss, validation_loss, learning_rate, rank):
    # Log metrics to TensorBoard if on the main process (rank 0)
    if rank == 0 and tb_writer is not None:
        tb_writer.add_scalar('Loss/train', train_loss, epoch)
        tb_writer.add_scalar('Loss/validation', validation_loss, epoch)
        tb_writer.add_scalar('Learning Rate', learning_rate, epoch)

######################################################################################################

## Save/Load Model ##
######################################################################################################
def save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir='checkpoints'):
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),  # Model weights
        'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
        'epoch': epoch,  # Current epoch
        'best_val_loss': best_val_loss  # Best validation loss
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch}_val_loss_{best_val_loss:.4f}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved at epoch {epoch} with val loss: {best_val_loss:.4f}")

def load_checkpoint(model, optimizer, checkpoint_path=''):
    # Load a checkpoint if the path exists
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from epoch {epoch} with val loss {best_val_loss:.4f}")
        return epoch, best_val_loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf')  # Return initial state if no checkpoint is found

def get_latest_checkpoint(checkpoint_dir):
    # Get the latest checkpoint from the directory
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None  # Return None if the directory does not exist
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if not checkpoints:
        print("No checkpoint files found.")
        return None  # Return None if no checkpoints are found

    # Extract epoch and validation loss information from filenames
    epoch_checkpoints = [(int(re.search(r'epoch_(\d+)', f).group(1)), f) for f in checkpoints]

    # Sort by epoch and find the latest checkpoint
    latest_checkpoint = max(epoch_checkpoints, key=lambda x: x[0])[1]
    
    return os.path.join(checkpoint_dir, latest_checkpoint)

######################################################################################################

## Check Devices Status ##
######################################################################################################
def check_device():
    # Check if GPU is available and print the device details
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        print("No GPU available. Please check your setup.")

def check_multi_gpu(world_size):
    # Verify if the required number of GPUs for distributed training is available
    num_gpus = torch.cuda.device_count()
    if num_gpus >= world_size:
        print(f"Using {world_size} GPUs for distributed training.")
    else:
        print(f"Warning: Expected {world_size} GPUs, but only {num_gpus} are available.")
        print("Please check your hardware setup or adjust the world_size.")

def check_mixed_precision():
    # Check if mixed precision training is enabled
    scaler = GradScaler()
    if isinstance(scaler, GradScaler):
        print("Mixed precision training is enabled (AMP).")
    else:
        print("Mixed precision training is not enabled.")

def check_ddp_usage(model, rank):
    # Verify if the model is correctly using DistributedDataParallel (DDP)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        print(f"Model is correctly wrapped with DistributedDataParallel on rank {rank}.")
    else:
        print(f"Warning: Model is not using DistributedDataParallel on rank {rank}.")

def print_gpu_memory_usage():
    # Print the GPU memory usage for each available GPU
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Allocated Memory: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"  Reserved Memory: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

######################################################################################################

#Setting_seed
######################################################################################################
def set_seed(seed):
    """
    Set the random seed for reproducibility in all necessary libraries and modules.

    Args:
        seed (int): The seed value to set for reproducibility.
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (single GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (all GPUs for multi-GPU DDP)

    # Ensure deterministic behavior for CuDNN (reproducibility vs. speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False