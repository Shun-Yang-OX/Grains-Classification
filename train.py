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
import utils

def train_ddp(rank, world_size, data_dir, Result_folder, batch_size, num_epochs, seed):
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

    utils.check_device()
    utils.check_multi_gpu(world_size)
    utils.check_mixed_precision()

    data_loader_train, data_loader_validation, data_test = dataset.create_data_loaders_ddp(data_dir, batch_size)

    model = Model.build_resnet152_for_xray(num_classes=2, freeze_backbone=True).to(device)
    model = DDP(model, device_ids=[rank])

    utils.check_ddp_usage(model, rank)

    optimizer = Model.configure_optimizer(model,train_only_classifier=True)
    total_iters = len(data_loader_train) * num_epochs
    warmup_iters = len(data_loader_train)
    scheduler = Model.initialize_scheduler(optimizer, warmup_iters, total_iters)
    scaler = GradScaler("cuda")


    checkpoint_dir = os.path.join(Result_folder, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        if rank == 0:
            os.makedirs(checkpoint_dir)
            print(f"Created checkpoint directory at {checkpoint_dir}")

    checkpoint_path = utils.get_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        start_epoch, best_val_loss = utils.load_checkpoint(model, optimizer, checkpoint_path)
        if rank == 0:
            print(f"Checkpoint found. Resuming training from epoch {start_epoch}.")
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        if rank == 0:
            print("No checkpoint found. Starting training from scratch.")

    global_step = 0
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        if epoch % 10 == 0:
            utils.print_gpu_memory_usage()

        train_loss, val_loss, global_step = Engine.train_and_validate_one_epoch_ddp(
            model, data_loader_train, data_loader_validation, data_loader_train.sampler, optimizer, scaler, device, epoch, rank, scheduler, tb_writer, global_step
        )

        learning_rate = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - start_time

        utils.log_metrics_to_file(epoch, train_loss, val_loss, learning_rate, epoch_duration, rank)

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if rank == 0:
                utils.save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir)
        elif epoch % 10 == 0 and rank == 0:
            utils.save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir)

    Engine.cleanup_ddp()
    if rank == 0 and tb_writer is not None:
        tb_writer.close()


def main_ddp(world_size, data_dir, Result_folder, batch_size, num_epochs, seed):
    mp.spawn(
        train_ddp,
        args=(world_size,data_dir, Result_folder, batch_size, num_epochs, seed),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    DATA_DIR = r'/home/shun/Project/Grains-Classification/Dataset/classifier_accuracy_test_final'
    RESULT_FOLDER = r'/home/shun/Project/Grains-Classification/Result_test/ResNet_frozen'

    world_size = 4  # Number of GPUs to use
    batch_size = 4
    num_epochs = 15
    seed = 10086

    main_ddp(world_size, DATA_DIR, RESULT_FOLDER, batch_size, num_epochs, seed)
