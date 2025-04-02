import os
import glob
import time
import torch
import wandb
import torch.multiprocessing as mp

from torch import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# Import custom modules
import Model
import dataset
import dataset_3class
import Engine
import utils_setup  # Assuming this is the file containing helper functions such as setup_logging, save_checkpoint, load_checkpoint, etc.
from dotenv import load_dotenv


# Load environment variables from a .env file
load_dotenv()

source_dir = os.getenv("WORKSPACE")
if source_dir is None:
    raise ValueError("Please set the WORKSPACE environment variable.")

def train_ddp(rank, world_size, data_dir, Result_folder, batch_size, num_epochs, seed, wandb_project, run_name):
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
        wandb_project (str): Project name for Weights & Biases.
        run_name (str): Run name for Weights & Biases.
    """
    # ------------------ 1) Set the random seed ------------------ #
    utils_setup.set_seed(seed + rank)

    # ------------------ 2) Initialize the distributed environment ------------------ #
    Engine.setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Only the main process creates the result directory
    if rank == 0 and not os.path.exists(Result_folder):
        os.makedirs(Result_folder)
        print(f"Created result directory at {Result_folder}")

    # Set up log directory & initialize wandb (only in main process)
    log_dir = os.path.join(Result_folder, 'logs')
    run = utils_setup.setup_logging(
        log_dir=log_dir,
        rank=rank,
        wandb_project=wandb_project,
        run_name=run_name
    )

    # Check device, DDP usage, etc.
    utils_setup.check_device()
    utils_setup.check_multi_gpu(world_size)
    utils_setup.check_mixed_precision()

    # ------------------ 3) Create data loaders (DDP) ------------------ #
    data_loader_train, data_loader_validation = dataset_3class.create_data_loaders_ddp(data_dir, batch_size)

    # ------------------ 4) Build model & wrap with distributed parallel ------------------ #
    model = Model.build_resnet152_for_xray(num_classes=3, freeze_backbone=True).to(device)
    model = DDP(model, device_ids=[rank])
    utils_setup.check_ddp_usage(model, rank)

    # ------------------ 5) Configure optimizer & scheduler ------------------ #
    optimizer = Model.configure_optimizer(model, train_only_classifier=True)
    total_iters = len(data_loader_train) * num_epochs
    warmup_iters = len(data_loader_train)
    scheduler = Model.initialize_scheduler(optimizer, warmup_iters, total_iters)

    # ------------------ 6) Mixed precision training related ------------------ #
    scaler = GradScaler("cuda")

    # ------------------ 7) Try loading the latest checkpoint ------------------ #
    checkpoint_dir = os.path.join(Result_folder, 'checkpoints')
    if rank == 0 and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory at {checkpoint_dir}")

    checkpoint_path = utils_setup.get_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        start_epoch, best_val_loss = utils_setup.load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
            scheduler=scheduler  # Let scheduler resume synchronously
        )
        if rank == 0:
            print(f"Checkpoint found. Resuming training from epoch {start_epoch}.")
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        if rank == 0:
            print("No checkpoint found. Starting training from scratch.")

    # ------------------ 8) Set wandb watch in the main process ------------------ #
    if rank == 0 and run is not None:
        wandb.watch(model.module if hasattr(model, "module") else model, log="all", log_freq=100)

    global_step = 0

    # ------------------ 9) Training loop ------------------ #
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        # Print GPU memory usage every 10 epochs
        if epoch % 10 == 0:
            utils_setup.print_gpu_memory_usage()

        # Training & validation for a single epoch
        train_loss, val_loss, global_step = Engine.train_and_validate_one_epoch_ddp(
            model=model,
            data_loader_train=data_loader_train,
            data_loader_val=data_loader_validation,
            train_sampler=data_loader_train.sampler,  # Ensure random sampling for DDP
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            rank=rank,
            scheduler=scheduler,
            experiment=run,
            global_step=global_step
        )

        learning_rate = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - start_time

        # Logging (file + wandb)
        utils_setup.log_metrics_to_file(epoch, train_loss, val_loss, learning_rate, epoch_duration, rank)
        utils_setup.log_metrics_to_wandb(
            run,
            epoch,
            train_loss=train_loss,
            validation_loss=val_loss,
            learning_rate=learning_rate,
            duration=epoch_duration,
            rank=rank
        )

        # ------------------ 10) Checkpoint saving logic ------------------ #
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if rank == 0:
                utils_setup.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    checkpoint_dir=checkpoint_dir,
                    scheduler=scheduler  # Save scheduler state
                )
        elif epoch % 10 == 0 and rank == 0:
            # Save checkpoint every 10 epochs
            utils_setup.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                checkpoint_dir=checkpoint_dir,
                scheduler=scheduler
            )

    # ------------------ 11) Training completion handling ------------------ #
    Engine.cleanup_ddp()

    # Only the main process handles wandb artifacts & run finish
    if rank == 0 and run is not None:
        best_checkpoint_path = utils_setup.get_latest_checkpoint(os.path.join(Result_folder, 'checkpoints'))
        if best_checkpoint_path is not None:
            artifact = wandb.Artifact("best_model", type="model", description="Best model checkpoint")
            artifact.add_file(best_checkpoint_path)
            run.log_artifact(artifact)
            print("Best model artifact uploaded.")

            # Upload all .py files
            code_artifact = wandb.Artifact("source_code", type="code", description="All Python source files from the project")
            for py_file in glob.glob(os.path.join(source_dir, "**", "*.py"), recursive=True):
                code_artifact.add_file(py_file)
            run.log_artifact(code_artifact)
            print("Source code artifact uploaded.")

        run.finish()

def main_ddp(world_size, data_dir, Result_folder, batch_size, num_epochs, seed, wandb_project, RUNNAME=None):
    """
    Initialize and start the Distributed Data Parallel (DDP) training process.
    """
    mp.spawn(
        train_ddp,
        args=(world_size, data_dir, Result_folder, batch_size, num_epochs, seed, wandb_project, RUNNAME),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # ------------------ 12) Read basic configuration ------------------ #
    DATA_DIR = r'/home/shun/Project/Grains-Classification/Dataset/dataset_v1/Data_input'
    RESULT_FOLDER = r'/home/shun/Project/Grains-Classification/Result_2025.2.22/ResNet_3_class_frozen'

    world_size = 4
    batch_size = 32
    num_epochs = 35
    seed = 10086

    WANDB_PROJECT = os.getenv("WANDB_PROJECT")
    RUNNAME = os.getenv("RUNMANE")  # Note: variable spelling

    if WANDB_PROJECT is None:
        raise ValueError("Please set the WANDB_PROJECT environment variable.")

    # ------------------ 13) Start DDP training ------------------ #
    main_ddp(world_size, DATA_DIR, RESULT_FOLDER, batch_size, num_epochs, seed, WANDB_PROJECT, RUNNAME)
