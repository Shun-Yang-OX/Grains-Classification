import os
import math
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms

class GrayToRGB:
    """Converter to transform a single-channel grayscale image into a three-channel RGB image."""
    def __call__(self, image):
        return image.convert('RGB')

class CustomXrayDataset(Dataset):
    def __init__(self, data_dir, phase='train', expected_size=(1200, 1200)):
        """
        Custom dataset for X-ray images that balances classes and applies data augmentation.

        Args:
            data_dir (str): Root directory containing class subdirectories with images.
            phase (str): One of 'train', 'validation', or 'test'. Affects augmentation usage.
            expected_size (tuple): Desired output size of images.
        """
        self.data_dir = data_dir
        self.phase = phase.lower()
        self.expected_size = expected_size

        # Mapping from class names to numeric labels
        self.label_map = {"equiax": 0, "columnar": 1}
        self.images = []
        self.labels = []
        self.image_transforms = []

        allowed_extensions = {".jpeg", ".jpg", ".png"}
        class_images = {}

        # Collect image paths and corresponding labels for each class
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path) and class_name in self.label_map:
                label = self.label_map[class_name]
                img_files = [
                    os.path.join(class_path, img_file)
                    for img_file in os.listdir(class_path)
                    if os.path.splitext(img_file)[1].lower() in allowed_extensions
                ]
                class_images[label] = img_files

        # Compute the maximum number of samples per class
        max_samples_per_class = 2 * max(len(class_images[label]) for label in class_images)

        # Initialize balanced lists for images, labels, and transform types
        balanced_images = []
        balanced_labels = []
        balanced_transforms = []

        for label, img_list in class_images.items():
            num_images = len(img_list)
            # Calculate the number of duplications needed to balance classes
            num_duplicates = math.ceil(max_samples_per_class / (2 * num_images))
            total_samples = 0
            for img_path in img_list:
                for _ in range(num_duplicates):
                    # Add original image entry
                    balanced_images.append(img_path)
                    balanced_labels.append(label)
                    balanced_transforms.append('original')

                    # Add augmented image entry
                    balanced_images.append(img_path)
                    balanced_labels.append(label)
                    balanced_transforms.append('augmented')
                    total_samples += 2
                    # Stop if reached maximum samples per class for this label
                    if total_samples >= max_samples_per_class:
                        break
                if total_samples >= max_samples_per_class:
                    break

        # Assign the balanced lists to the dataset's attributes
        self.images = balanced_images
        self.labels = balanced_labels
        self.image_transforms = balanced_transforms

        # Define basic transformation pipeline (for original images without augmentation)
        self.basic_transform = transforms.Compose([
            GrayToRGB(),
            transforms.Resize(self.expected_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Define augmentation transformation pipeline (applied during training)
        self.augmentation_transform = transforms.Compose([
            GrayToRGB(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([
                transforms.RandomRotation(degrees=[0, 0]),
                transforms.RandomRotation(degrees=[90, 90]),
                transforms.RandomRotation(degrees=[180, 180]),
                transforms.RandomRotation(degrees=[270, 270])
            ]),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomCrop(self.expected_size),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve an image and its label at the specified index, applying the appropriate transformation.

        Args:
            idx (int): Index of the desired sample.

        Returns:
            image (Tensor): Transformed image tensor.
            label (int): Corresponding label for the image.
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        transform_type = self.image_transforms[idx]

        # Load image using PIL
        image = Image.open(img_path)

        # Apply appropriate transformations based on the type ('original' or 'augmented')
        if transform_type == 'original':
            # Apply basic transformation (no augmentation)
            image = self.basic_transform(image)
        elif transform_type == 'augmented':
            # Apply augmentation transformation if in training phase
            if self.phase == 'train':
                image = self.augmentation_transform(image)
            else:
                # If not training phase, use basic transformation
                image = self.basic_transform(image)

        return image, label

def create_data_loaders_ddp(data_dir, batch_size=32, num_workers=8):
    """
    Create DataLoaders for training, validation, and testing using DistributedSampler for DDP.

    Args:
        data_dir (str): Root directory containing 'train', 'validation', and 'test' subdirectories.
        batch_size (int): Batch size per GPU.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        Tuple of DataLoaders: (train_loader, val_loader, test_loader)
    """
    # Create dataset objects for each phase
    train_data = CustomXrayDataset(data_dir=os.path.join(data_dir, 'train'),
                                   phase='train')
    val_data = CustomXrayDataset(data_dir=os.path.join(data_dir, 'validation'),
                                 phase='validation')
    test_data = CustomXrayDataset(data_dir=os.path.join(data_dir, 'test'),
                                  phase='test')

    # Use DistributedSampler to split data across GPUs/processes
    train_sampler = DistributedSampler(train_data, shuffle=True)
    val_sampler = DistributedSampler(val_data, shuffle=False)
    test_sampler = DistributedSampler(test_data, shuffle=False)

    # Create DataLoaders with the corresponding DistributedSamplers
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=True, drop_last=True)

    return train_loader, val_loader, test_loader
