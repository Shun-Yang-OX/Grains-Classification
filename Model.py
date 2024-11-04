import torch
import timm
import torchvision.models as models
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


def build_resnet152_for_xray(num_classes, pretrained=True):
    # Load the pretrained ResNet-152 model
    model = models.resnet152(pretrained=pretrained)
    
    # Modify the first convolution layer to accept single-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify the fully connected layer to set the output number of classes to `num_classes`
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

import torch
from torchvision import models
import torch.nn as nn


def build_swin_transformer_model(model_name='swin_large_patch4_window12_384_in22k',
                                 num_classes=2,
                                 img_size=1200,
                                 pretrained=True,
                                 in_chans=1
                                 ):
    """
    Build and return a Swin Transformer model for classification tasks.

    Args:
        model_name (str): Name of the Swin Transformer model in the `timm` library.
                          Default is 'swin_large_patch4_window12_384_in22k'.
        num_classes (int): Number of output classes. Set to 2 for binary classification tasks.
        img_size (int or tuple): Input image size. If int, the image will be resized to (img_size, img_size).
        pretrained (bool): Whether to load pretrained weights. Default is True.

    Returns:
        model (torch.nn.Module): The constructed Swin Transformer model.
    """
    # Create the model
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, img_size=img_size, in_chans=in_chans)
        print(f"Successfully loaded model: {model_name}, pretrained: {pretrained}")
    except Exception as e:
        raise ValueError(f"Unable to create model '{model_name}'. Please check if the model name is correct and if the 'timm' library supports this model. Error details: {e}")
    
    return model


def configure_sgd_optimizer(model, learning_rate=1e-3, momentum=0.9, weight_decay=1e-4):
    # Define the SGD optimizer
    optimizer = SGD(
        model.parameters(), 
        lr=learning_rate, 
        momentum=momentum, 
        weight_decay=weight_decay
    )
    return optimizer

# Initialize learning rate scheduler
def initialize_scheduler(optimizer, warmup_iters, total_iters):
    """
    Initialize a learning rate scheduler with a warm-up phase followed by a cosine annealing phase.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be scheduled.
        warmup_iters (int): The number of iterations for the warm-up phase.
        total_iters (int): The total number of iterations for training.
        
    Returns:
        scheduler (torch.optim.lr_scheduler.SequentialLR): The composed learning rate scheduler.
    """
    # Get the initial learning rate (base_lr)
    base_lr = optimizer.param_groups[0]['lr']
    
    # Define the learning rate at the start of the warm-up
    start_lr = base_lr * 0.01  # Assume the starting learning rate is 1% of base_lr
    
    # Calculate start_factor
    start_factor = start_lr / base_lr  # 0.01
    end_factor = 1.0  # At the end of the warm-up, the learning rate reaches base_lr

    # Define LinearLR as the warm-up scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=warmup_iters
    )
    
    # Define CosineAnnealingLR for the subsequent learning rate schedule
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_iters - warmup_iters),
        eta_min=1e-5  # Set a small non-zero learning rate at the end
    )
    
    # Use SequentialLR to combine both schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iters]
    )
    
    return scheduler
