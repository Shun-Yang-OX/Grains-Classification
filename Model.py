import torch
import timm
import torchvision.models as models
from torch import nn
from torch.optim import SGD
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


import torch.nn as nn
from torchvision import models

def build_resnet152_for_xray(num_classes, pretrained=True, freeze_backbone=False):
    """
    Build a ResNet-152 model for X-ray image classification.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use a pretrained ResNet-152.
        freeze_backbone (bool): If True, freeze all layers except the classification head.

    Returns:
        nn.Module: The modified ResNet-152 model.
    """
    # Load the pretrained ResNet-152 model
    model = models.resnet152(pretrained=pretrained)
    
    # Modify the fully connected layer to set the output number of classes to `num_classes`
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    if freeze_backbone:
        # Freeze all layers except the classification head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model



def build_swin_transformer_model(model_name='swin_large_patch4_window12_384_in22k',
                                 num_classes=2,
                                 img_size=1200,
                                 pretrained=True,
                                 in_chans=3,
                                 freeze_backbone=False):
    """
    Build and return a Swin Transformer model for classification tasks.

    Args:
        model_name (str): Name of the Swin Transformer model in the timm library.
        num_classes (int): Number of output classes. Set to 2 for binary classification tasks.
        img_size (int or tuple): Input image size. If int, the image will be resized to (img_size, img_size).
        pretrained (bool): Whether to load pretrained weights. Default is True.
        in_chans (int): Number of input channels. Default is 3 (RGB images).
        freeze_backbone (bool): Whether to freeze the backbone parameters. Default is False.

    Returns:
        model (torch.nn.Module): The constructed Swin Transformer model.
    """
    # Create the model
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes,
                                  img_size=img_size, in_chans=in_chans)
        print(f"Successfully loaded model: {model_name}, pretrained: {pretrained}")
    except Exception as e:
        raise ValueError(f"Unable to create model '{model_name}'. Please check if the model name is correct and if the 'timm' library supports this model. Error details: {e}")
    
    # Decide whether to freeze the backbone based on the freeze_backbone parameter
    if freeze_backbone:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the parameters of the classification head
        if hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
            print("Backbone parameters have been frozen. Only the classification head will be trained.")
        else:
            # If the model does not have a 'head' attribute, manually specify the location of the classification head
            print("The model does not have a direct 'head' attribute. Here are the parameter names:")
            for name, param in model.named_parameters():
                print(name)
            raise ValueError("Please modify the code to unfreeze the classification head parameters based on the parameter names above.")
    else:
        print("Training the entire model, including the backbone and classification head.")
    
    return model


def configure_optimizer(model, learning_rate=3e-5, momentum=0.9, weight_decay=1e-4, train_only_classifier=False):
    """
    Configure the SGD optimizer.

    Args:
        model (torch.nn.Module): The model to be trained.
        learning_rate (float): Learning rate.
        momentum (float): Momentum factor.
        weight_decay (float): Weight decay (L2 penalty).
        train_only_classifier (bool): Whether to train only the classification head. Default is False.

    Returns:
        optimizer (torch.optim.Optimizer): The configured optimizer.
    """
    if train_only_classifier:
        # Collect trainable parameters
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable parameter: {name}")
                trainable_params.append(param)
        print("Optimizer is configured to train only the parameters of the classification head.")
    else:
        # Include all parameters of the model
        trainable_params = model.parameters()
        print("Optimizer is configured to train all parameters of the model.")
    
    # Define the SGD optimizer
    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
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
