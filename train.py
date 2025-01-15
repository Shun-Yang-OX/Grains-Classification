import os
import time
import torch
import torch.multiprocessing as mp

from torch import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# Import custom modules
import Model
import dataset
import Engine
import utils_setup

# Import Environment Variables
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file


def train_ddp(rank, world_size, data_dir, Result_folder, batch_size, num_epochs, seed, comet_api_key, comet_project_name, comet_workspace):
    """
    Train the model using Distributed Data Parallel (DDP).

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes (typically equal to the number of GPUs).
        data_dir (str): Directory containing the dataset.
        Result_folder (str): Directory to save results and checkpoints.
        batch_size (int): Number of samples per batch.
        num_epochs (int): Total number of training epochs.
        seed (int): Base seed for reproducibility.
        comet_api_key (str): API key for Comet.ml.
        comet_project_name (str): Project name for Comet.ml.
        comet_workspace (str): Workspace name for Comet.ml.
    """
    # Set the random seed for reproducibility, adjusted by the process rank
    utils_setup.set_seed(seed + rank)
    
    # Initialize distributed training environment
    Engine.setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')  # Assign a specific GPU to this process
    
    # Create the result directory if it doesn't exist (only by the main process)
    if not os.path.exists(Result_folder):
        if rank == 0:
            os.makedirs(Result_folder)
            print(f"Created result directory at {Result_folder}")
    
    # Define the logging directory within the result folder
    log_dir = os.path.join(Result_folder, 'logs')
    
    # Initialize logging and Comet.ml experiment (only for the main process)
    experiment = utils_setup.setup_logging(
        log_dir=log_dir,
        rank=rank,
        comet_api_key=comet_api_key,
        comet_project_name=comet_project_name,
        comet_workspace=comet_workspace
    )
    
    # Perform device and environment checks to ensure everything is set up correctly
    utils_setup.check_device()
    utils_setup.check_multi_gpu(world_size)
    utils_setup.check_mixed_precision()
    
    # Create data loaders for training, validation, and testing using DDP
    data_loader_train, data_loader_validation, data_test = dataset.create_data_loaders_ddp(data_dir, batch_size)
    
    # Build the ResNet152 model tailored for X-ray classification and move it to the designated device
    model = Model.build_resnet152_for_xray(num_classes=2, freeze_backbone=True).to(device)
    
    # Wrap the model with DistributedDataParallel to handle gradient synchronization across processes
    model = DDP(model, device_ids=[rank])
    
    # Verify that the model is correctly wrapped with DDP
    utils_setup.check_ddp_usage(model, rank)
    
    # Configure the optimizer, specifying whether to train only the classifier layers
    optimizer = Model.configure_optimizer(model, train_only_classifier=True)
    
    # Calculate total and warmup iterations for the learning rate scheduler
    total_iters = len(data_loader_train) * num_epochs
    warmup_iters = len(data_loader_train)
    
    # Initialize the learning rate scheduler
    scheduler = Model.initialize_scheduler(optimizer, warmup_iters, total_iters)
    
    # Initialize GradScaler for mixed precision training (Automatic Mixed Precision - AMP)
    scaler = GradScaler("cuda")
    
    # Define the checkpoint directory within the result folder
    checkpoint_dir = os.path.join(Result_folder, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        if rank == 0:
            os.makedirs(checkpoint_dir)
            print(f"Created checkpoint directory at {checkpoint_dir}")
    
    # Attempt to load the latest checkpoint to resume training
    checkpoint_path = utils_setup.get_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        start_epoch, best_val_loss = utils_setup.load_checkpoint(model, optimizer, checkpoint_path)
        if rank == 0:
            print(f"Checkpoint found. Resuming training from epoch {start_epoch}.")
    else:
        # If no checkpoint is found, start training from scratch
        start_epoch = 0
        best_val_loss = float('inf')
        if rank == 0:
            print("No checkpoint found. Starting training from scratch.")
    
    global_step = 0  # Initialize the global step counter
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()  # Record the start time of the epoch
        
        # Periodically print GPU memory usage for monitoring (every 10 epochs)
        if epoch % 10 == 0:
            utils_setup.print_gpu_memory_usage()
        
        # Train and validate the model for one epoch
        train_loss, val_loss, global_step = Engine.train_and_validate_one_epoch_ddp(
            model=model,
            data_loader_train=data_loader_train,
            data_loader_val=data_loader_validation,
            train_sampler=data_loader_train.sampler,  # Ensure proper shuffling for DDP
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            rank=rank,
            scheduler=scheduler,
            experiment=experiment,  # Pass the Comet.ml experiment object for logging
            global_step=global_step
        )
        
        # Retrieve the current learning rate from the optimizer
        learning_rate = optimizer.param_groups[0]['lr']
        
        # Calculate the duration of the epoch
        epoch_duration = time.time() - start_time
        
        # Log the training and validation metrics to the log file
        utils_setup.log_metrics_to_file(epoch, train_loss, val_loss, learning_rate, epoch_duration, rank)
        
        # Log the metrics to Comet.ml for experiment tracking and visualization
        utils_setup.log_metrics_to_comet(
            experiment,
            epoch,
            train_loss=train_loss,
            validation_loss=val_loss,
            learning_rate=learning_rate,
            duration=epoch_duration,
            rank=rank
        )
        
        # Check if the current validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update the best validation loss
            if rank == 0:
                # Save the model checkpoint as it has achieved a new best validation loss
                utils_setup.save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir)
        elif epoch % 10 == 0 and rank == 0:
            # Additionally save checkpoints every 10 epochs regardless of validation loss
            utils_setup.save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir)
    
    # Clean up the distributed training environment after all epochs are completed
    Engine.cleanup_ddp()
    
    # End the Comet.ml experiment to finalize logging and metrics (only by the main process)
    if rank == 0 and experiment is not None:
        experiment.end()


def main_ddp(world_size, data_dir, Result_folder, batch_size, num_epochs, seed, comet_api_key, comet_project_name, comet_workspace):
    """
    Initialize and start the Distributed Data Parallel (DDP) training process.

    Args:
        world_size (int): Total number of processes (typically equal to the number of GPUs).
        data_dir (str): Directory containing the dataset.
        Result_folder (str): Directory to save results and checkpoints.
        batch_size (int): Number of samples per batch.
        num_epochs (int): Total number of training epochs.
        seed (int): Base seed for reproducibility.
        comet_api_key (str): API key for Comet.ml.
        comet_project_name (str): Project name for Comet.ml.
        comet_workspace (str): Workspace name for Comet.ml.
    """
    # Spawn multiple processes for distributed training, each running the train_ddp function
    mp.spawn(
        train_ddp,
        args=(world_size, data_dir, Result_folder, batch_size, num_epochs, seed, comet_api_key, comet_project_name, comet_workspace),
        nprocs=world_size,  # Number of processes to spawn (typically equal to the number of GPUs)
        join=True  # Wait for all processes to finish
    )


if __name__ == "__main__":
    # Define the directory paths for data and results
    DATA_DIR = r'/home/shun/Project/Grains-Classification/Dataset/Data_input'
    RESULT_FOLDER = r'/home/shun/Project/Grains-Classification/Result_test_2025.01/ResNet_frozen'
    
    # Set training parameters
    world_size = 4  # Number of GPUs to use (and hence the number of processes)
    batch_size = 4  # Number of samples per batch per GPU
    num_epochs = 20  # Total number of training epochs
    seed = 10086  # Base seed for reproducibility
    
    # Read Comet.ml configuration from environment variables
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    COMET_PROJECT_NAME = os.getenv("COMET_PROJECT_NAME")
    COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")
    
    # Ensure that all necessary Comet.ml environment variables are set
    if COMET_API_KEY is None or COMET_PROJECT_NAME is None or COMET_WORKSPACE is None:
        raise ValueError("Please set the COMET_API_KEY, COMET_PROJECT_NAME, and COMET_WORKSPACE environment variables.")
    
    # Start the DDP training process
    main_ddp(world_size, DATA_DIR, RESULT_FOLDER, batch_size, num_epochs, seed, COMET_API_KEY, COMET_PROJECT_NAME, COMET_WORKSPACE)
