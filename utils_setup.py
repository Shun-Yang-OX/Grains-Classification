from torch.amp import GradScaler
import logging
import random
import numpy as np
import torch
import re
import os
import wandb  

## Logger ##
######################################################################################################
def setup_logging(log_dir, rank, wandb_project=None, run_name=None):
    """
    Set up logging. For the main process (rank=0), configure file logging and initialize a wandb run.

    Parameters:
    - log_dir (str): Log directory.
    - rank (int): Process rank.
    - wandb_project (str, optional): Project name on Weights & Biases.
    - run_name (str, optional): Run name on Weights & Biases.
    """
    run = None
    if rank == 0:
        # Set up file logging
        log_file = os.path.join(log_dir, f'train_rank_{rank}.log')
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        logging.info("Logging is set up.")

        # Initialize wandb run
        run = wandb.init(
            project=wandb_project,
            name=run_name,
            reinit=True  # Allow reinitialization if needed
        )
        logging.info("Weights & Biases run is initialized.")
    return run

def log_metrics_to_file(epoch, train_loss, validation_loss, learning_rate, duration, rank):
    """
    Log training metrics to a log file (main process only).

    Parameters:
    - epoch (int): Current epoch.
    - train_loss (float): Training loss.
    - validation_loss (float): Validation loss.
    - learning_rate (float): Current learning rate.
    - duration (float): Duration of this epoch (seconds).
    - rank (int): Process rank.
    """
    if rank == 0:
        logging.info(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, LR: {learning_rate:.6f}, Duration: {duration:.2f}s")

def log_metrics_to_wandb(run, global_step, train_loss=None, validation_loss=None, learning_rate=None, duration=None, rank=0):
    """
    Log training metrics to Weights & Biases (main process only).

    Parameters:
    - run (Run): wandb Run object.
    - global_step (int): Global step.
    - train_loss (float, optional): Training loss.
    - validation_loss (float, optional): Validation loss.
    - learning_rate (float, optional): Current learning rate.
    - duration (float, optional): Duration of this epoch (seconds).
    - rank (int): Process rank.
    """
    if rank == 0 and run is not None:
        metrics = {"global_step": global_step}
        if train_loss is not None:
            metrics["Loss/train"] = train_loss
        if validation_loss is not None:
            metrics["Loss/validation"] = validation_loss
        if learning_rate is not None:
            metrics["Learning Rate"] = learning_rate
        if duration is not None:
            metrics["Duration (s)"] = duration

        run.log(metrics, step=global_step)

######################################################################################################

## Early Stop ##
######################################################################################################
class EarlyStopping:
    """
    Stop training early if the validation loss does not improve within a given patience.

    Parameters:
        patience (int): Maximum number of consecutive epochs with no improvement before stopping.
        verbose (bool): If True, print messages when saving the model.
        delta (float): Minimum improvement threshold. Only if the validation loss decreases by more than this value is it considered an improvement.
        path (str): File path to save the best model.
        trace_func (function): Function for printing log messages, default is print.
    """
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save the model parameters when the validation loss decreases.
        """
        if self.verbose:
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

######################################################################################################

## Save/Load Model ##
######################################################################################################
def save_checkpoint(model, optimizer, epoch, best_val_loss,
                    checkpoint_dir='checkpoints',
                    scheduler=None):
    """
    Save a checkpoint including model state, optimizer state, epoch, best_val_loss,
    and (optionally) the scheduler state.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f'best_model_epoch_{epoch}_val_loss_{best_val_loss:.4f}.pth'
    )

    # Package core information into the checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
    }

    # If a scheduler is provided, save its state_dict
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved at epoch {epoch} with val loss: {best_val_loss:.4f}")


def load_checkpoint(model, optimizer, checkpoint_path='', scheduler=None):
    """
    Load a checkpoint and restore the model, optimizer, epoch, best_val_loss,
    and (optionally) the scheduler state.
    """
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']

        # If the checkpoint contains scheduler_state_dict and a scheduler is provided, restore it
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from epoch {epoch} with val loss {best_val_loss:.4f}")
        return epoch, best_val_loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf')

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

## Setting Seed ##
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
