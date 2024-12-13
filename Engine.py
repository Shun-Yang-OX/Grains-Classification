import torch
import os
import torch.distributed as dist
from torch import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F
import utils
from collections import deque

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def train_and_validate_one_epoch_ddp(model, data_loader_train, data_loader_val, train_sampler, optimizer, scaler, device, epoch, rank, scheduler, tb_writer=None, global_step=0):
    model.train()  # Switch to training mode
    train_sampler.set_epoch(epoch)
    
    total_steps_per_epoch = len(data_loader_train)
    total_train_loss = 0.0

    # 用于平滑loss的队列，最多存储最近35个step的loss
    recent_losses = deque(maxlen=20)

    # Create tqdm progress bar for the training set
    if rank == 0:
        train_progress_bar = tqdm(data_loader_train, desc=f"Rank {rank} | Epoch {epoch} [Training]", position=rank, leave=True)
    else:
        train_progress_bar = data_loader_train

    # -------------------- Training Loop --------------------
    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast("cuda"):
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()
        recent_losses.append(loss.item())  # 将本步的loss添加到队列中

        if scheduler:
            scheduler.step()

        if rank == 0:
            train_progress_bar.set_postfix(loss=loss.item())

        # 训练时更新global_step，并同步给所有进程
        if rank == 0:
            global_step += 1
        global_step_tensor = torch.tensor(global_step, dtype=torch.int64, device=device)
        dist.broadcast(global_step_tensor, src=0)
        global_step = global_step_tensor.item()

        # 每隔35个step对loss进行平滑后记录
        if rank == 0 and tb_writer is not None and (global_step % 20 == 0):
            # 对recent_losses求平均
            smoothed_loss = sum(recent_losses) / len(recent_losses)
            learning_rate = optimizer.param_groups[0]['lr']

            # 使用平滑后的loss进行记录
            utils.log_metrics_to_tensorboard(tb_writer, global_step, train_loss=smoothed_loss, learning_rate=learning_rate, rank=rank)

            # 记录其他分布信息
            probs = F.softmax(outputs, dim=1)
            pos_probs = probs[:, 1].detach().cpu().numpy()
            neg_probs = probs[:, 0].detach().cpu().numpy()

            tb_writer.add_histogram('Train/Output_Probabilities_PosClass', pos_probs, global_step)
            tb_writer.add_histogram('Train/Output_Probabilities_NegClass', neg_probs, global_step)

            logits = outputs.detach().cpu().numpy()
            tb_writer.add_histogram('Train/Logits_Class0', logits[:,0], global_step)
            tb_writer.add_histogram('Train/Logits_Class1', logits[:,1], global_step)

            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            tb_writer.add_histogram('Train/Predicted_Labels', preds, global_step)

    # 聚合训练loss
    total_train_loss_tensor = torch.tensor(total_train_loss, dtype=torch.float32, device=device)
    dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)
    total_train_loss_tensor = total_train_loss_tensor / dist.get_world_size()
    average_train_loss = total_train_loss_tensor.item() / len(data_loader_train)

    if rank == 0:
        print(f"Rank {rank} | Epoch [{epoch}] Training Completed | Average Train Loss (all GPUs): {average_train_loss:.4f}")

    # -------------------- Validation Loop --------------------
    # 不在验证阶段更新global_step
    model.eval()
    total_val_loss = 0.0

    if rank == 0:
        val_progress_bar = tqdm(data_loader_val, desc=f"Rank {rank} | Epoch {epoch} [Validation]", position=rank, leave=True)
    else:
        val_progress_bar = data_loader_val

    all_outputs = []
    all_labels = []

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

    # 聚合验证loss
    total_val_loss_tensor = torch.tensor(total_val_loss, dtype=torch.float32, device=device)
    dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
    total_val_loss_tensor = total_val_loss_tensor / dist.get_world_size()
    average_val_loss = total_val_loss_tensor.item() / len(data_loader_val)

    # 验证阶段只在epoch结束后记录一次
    if rank == 0 and tb_writer is not None:
        utils.log_metrics_to_tensorboard(tb_writer, global_step, validation_loss=average_val_loss, rank=rank)

        all_outputs = torch.cat(all_outputs, dim=0)
        probs = F.softmax(all_outputs, dim=1)
        pos_probs = probs[:, 1].numpy()
        neg_probs = probs[:, 0].numpy()

        tb_writer.add_histogram('Validation/Output_Probabilities_PosClass', pos_probs, global_step)
        tb_writer.add_histogram('Validation/Output_Probabilities_NegClass', neg_probs, global_step)

        logits = all_outputs.numpy()
        tb_writer.add_histogram('Validation/Logits_Class0', logits[:,0], global_step)
        tb_writer.add_histogram('Validation/Logits_Class1', logits[:,1], global_step)

        preds = torch.argmax(all_outputs, dim=1).numpy()
        tb_writer.add_histogram('Validation/Predicted_Labels', preds, global_step)

        print(f"Rank {rank} | Epoch [{epoch}] Validation Completed | Average Val Loss (all GPUs): {average_val_loss:.4f}")

    return average_train_loss, average_val_loss, global_step
