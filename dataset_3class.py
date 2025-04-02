import os
import math
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms

class GrayToRGB:
    """Convert a single-channel grayscale image to a three-channel RGB image."""
    def __call__(self, image):
        return image.convert('RGB')

class BottomCrop:
    """
    Crop a region of a specified size from the bottom of the image, horizontally centered.
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        width, height = img.size
        crop_width, crop_height = self.size
        left = (width - crop_width) // 2
        top = height - crop_height
        return img.crop((left, top, left + crop_width, top + crop_height))

class CustomXrayDataset(Dataset):
    def __init__(self, data_dir, phase='train', expected_size=(1056, 1056)):
        """
        Custom dataset for X-ray images that implements class balancing and data augmentation/single-crop evaluation.

        Args:
            data_dir (str): Root directory containing subfolders for each class.
            phase (str): 'train' or 'validation'.
            expected_size (tuple): Expected output size of augmentation, e.g., (1056, 1056).
        """
        self.data_dir = data_dir
        self.phase = phase.lower()
        self.expected_size = expected_size  # Uniformly set crop size to 1056x1056

        # Mapping from class names to numerical labels, retaining only three classes
        self.label_map = {
            "equiax": 0,
            "columnar": 1,
            "background": 2
        }

        self.images = []
        self.labels = []
        self.image_transforms = []

        allowed_extensions = {".jpeg", ".jpg", ".png"}
        class_images = {}

        # Traverse the data directory to collect image paths for each class
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

        # Calculate the maximum number of samples needed for balancing each class (multiplied by 2 to distinguish original and augmented samples)
        max_samples_per_class = 2 * max(len(class_images[label]) for label in class_images)

        balanced_images = []
        balanced_labels = []
        balanced_transforms = []

        for label, img_list in class_images.items():
            num_images = len(img_list)
            num_duplicates = math.ceil(max_samples_per_class / (2 * num_images))
            total_samples = 0
            for img_path in img_list:
                for _ in range(num_duplicates):
                    # Add original image
                    balanced_images.append(img_path)
                    balanced_labels.append(label)
                    balanced_transforms.append('original')
                    # Add augmented image
                    balanced_images.append(img_path)
                    balanced_labels.append(label)
                    balanced_transforms.append('augmented')
                    total_samples += 2
                    if total_samples >= max_samples_per_class:
                        break
                if total_samples >= max_samples_per_class:
                    break

        self.images = balanced_images
        self.labels = balanced_labels
        self.image_transforms = balanced_transforms

        # ------------------ Define preprocessing pipelines for different phases ------------------ #
        if self.phase == 'train':
            self.basic_transform = transforms.Compose([
                GrayToRGB(),
                transforms.RandomCrop(self.expected_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self.augmentation_transform = transforms.Compose([
                GrayToRGB(),
                transforms.RandomCrop(self.expected_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomChoice([
                    transforms.RandomRotation(degrees=[0, 0]),
                    transforms.RandomRotation(degrees=[90, 90]),
                    transforms.RandomRotation(degrees=[180, 180]),
                    transforms.RandomRotation(degrees=[270, 270])
                ]),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        elif self.phase == 'validation':
            self.single_crop_transform = transforms.Compose([
                GrayToRGB(),
                transforms.RandomCrop(self.expected_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Return the image and its label based on the index.
        During training, decide whether to use data augmentation based on the marker;
        during validation, directly use the single-crop preprocessing, returning the image as a tensor.
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        transform_type = self.image_transforms[idx]

        # Open the image using PIL
        image = Image.open(img_path)

        if self.phase == 'train':
            if transform_type == 'original':
                image = self.basic_transform(image)
            elif transform_type == 'augmented':
                image = self.augmentation_transform(image)
        elif self.phase == 'validation':
            image = self.single_crop_transform(image)
        return image, label

def create_data_loaders_ddp(data_dir, batch_size=32, num_workers=8):
    """
    Construct training and validation DataLoaders using DistributedSampler.

    Args:
        data_dir (str): Root directory containing 'train' and 'validation' subdirectories.
        batch_size (int): Batch size per GPU.
        num_workers (int): Number of subprocesses used for data loading.

    Returns:
        Tuple: (train_loader, val_loader)
    """
    train_data = CustomXrayDataset(
        data_dir=os.path.join(data_dir, 'train'),
        phase='train'
    )
    val_data = CustomXrayDataset(
        data_dir=os.path.join(data_dir, 'validation'),
        phase='validation'
    )

    train_sampler = DistributedSampler(train_data, shuffle=True)
    val_sampler = DistributedSampler(val_data, shuffle=False)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader
