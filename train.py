# ----------------------------------------
# Imports
# ----------------------------------------

import os
import time
import torch
import torch.multiprocessing as mp
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# Import custom modules
import Model
import dataset
import Engine
import utils

# ----------------------------------------
# Training Function
# ----------------------------------------

def train_ddp(rank, world_size, data_dir, Result_folder, batch_size, num_epochs, seed):
    """
    Function to train the model using Distributed Data Parallel (DDP) across multiple GPUs.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes (GPUs) participating in training.
        train_data_folder (str): Path to the training data folder.
        validation_folder (str): Path to the validation data folder.
        annotation_file (str): Path to the annotation file.
        Result_folder (str): Path to the folder where results and checkpoints will be saved.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs to train.
        n_augments (int): Number of data augmentations to apply.
    """
    # Initialize Distributed Data Parallel (DDP)
    utils.set_seed(seed + rank)
    Engine.setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Create Result folder if it does not exist
    if not os.path.exists(Result_folder):
        if rank == 0:
            os.makedirs(Result_folder)
            print(f"Created result directory at {Result_folder}")
    log_dir = os.path.join(Result_folder, 'logs')
    tb_writer = utils.setup_logging(log_dir=log_dir, rank=rank)

    # Check device and environment setup
    utils.check_device()
    utils.check_multi_gpu(world_size)
    utils.check_mixed_precision()

    # Load training and validation datasets
    data_loader_train, data_loader_validation, data_test = dataset.create_data_loaders_ddp(data_dir, batch_size)

    # Create the model and move it to the appropriate device
    model = Model.build_resnet152_for_xray(num_classes=2).to(device)

    # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[rank])

    # Check DDP usage
    utils.check_ddp_usage(model, rank)

    # Initialize optimizer, scheduler, and scaler for mixed-precision training
    optimizer = Model.configure_sgd_optimizer(model)
    total_iters = len(data_loader_train) * num_epochs
    warmup_iters = 500
    scheduler = Model.initialize_scheduler(optimizer, warmup_iters, total_iters)
    scaler = GradScaler()

    # Check if checkpoint directory exists, create if not
    checkpoint_dir = os.path.join(Result_folder, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        if rank == 0:
            os.makedirs(checkpoint_dir)
            print(f"Created checkpoint directory at {checkpoint_dir}")

    # Load from checkpoint if available
    checkpoint_path = utils.get_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        # If checkpoint exists, load model and optimizer state
        start_epoch, best_val_loss = utils.load_checkpoint(model, optimizer, checkpoint_path)
        if rank == 0:
            print(f"Checkpoint found. Resuming training from epoch {start_epoch}.")
    else:
        # If no checkpoint, start training from scratch
        start_epoch = 0
        best_val_loss = float('inf')
        if rank == 0:
            print("No checkpoint found. Starting training from scratch.")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        if epoch % 10 == 0:
            utils.print_gpu_memory_usage()

        # Train and validate for one epoch
        train_loss, val_loss = Engine.train_and_validate_one_epoch_ddp(
            model, data_loader_train, data_loader_validation, data_loader_train.sampler, optimizer, scaler, device, epoch, rank, scheduler
        )

        # Get learning rate and epoch duration
        learning_rate = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - start_time

        # Log metrics to file and TensorBoard
        utils.log_metrics_to_file(epoch, train_loss, val_loss, learning_rate, epoch_duration, rank)
        utils.log_metrics_to_tensorboard(tb_writer, epoch, train_loss, val_loss, learning_rate, rank)

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if rank == 0:  # Only save the model on the main process to avoid conflicts
                utils.save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir)
        elif epoch % 10 == 0 and rank == 0:
            # Save checkpoint every 10 epochs
            utils.save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir)

    # Training completed, cleanup DDP processes
    Engine.cleanup_ddp()
    tb_writer.close()

# ----------------------------------------
# Main Function
# ----------------------------------------

def main_ddp(world_size, data_dir, Result_folder, batch_size, num_epochs, seed):
    """
    Main function to initiate the distributed training using DDP.

    Args:
        world_size (int): Total number of processes (GPUs) participating in training.
        train_data_folder (str): Path to the training data folder.
        validation_folder (str): Path to the validation data folder.
        annotation_file (str): Path to the annotation file.
        Result_folder (str): Path to the folder where results and checkpoints will be saved.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs to train.
        n_augments (int): Number of data augmentations to apply.
    """
    # Launch multiple processes for DDP training
    mp.spawn(
        train_ddp,
        args=(world_size,data_dir, Result_folder, batch_size, num_epochs, seed),
        nprocs=world_size,
        join=True
    )

# ----------------------------------------
# Entry Point
# ----------------------------------------

if __name__ == "__main__":

    DATA_DIR = r'/home/shun/Project/Grains-Classification/Data'
    RESULT_FOLDER = r'/home/shun/Project/Grains-Classification/Result'

    # Configuration parameters
    world_size = 4  # Number of GPUs to use
    batch_size = 16
    num_epochs = 100
    seed = 10086

    # Start the distributed training
    main_ddp(world_size, DATA_DIR, RESULT_FOLDER, batch_size, num_epochs, seed)
