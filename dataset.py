import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms


class CustomXrayDataset(Dataset):
    def __init__(self, data_dir, phase='train', transform=None):
        """
        Custom X-ray image dataset class.
        Args:
            data_dir (str): Path to the data directory.
            phase (str): Dataset phase, 'train', 'validation' or 'test'.
            transform (callable, optional): Image transformation function.
        """
        self.data_dir = data_dir
        self.phase = phase.lower()
        self.transform = transform

        # Define mapping from class names to numbers
        self.label_map = {"equiax": 0, "columnar": 1}
        self.images = []
        self.labels = []

        # Define allowed image file extensions
        allowed_extensions = {".jpeg", ".jpg", ".png"}  # Add other extensions if needed

        # Collect image paths for each class
        class_images = {}  # {label: [img_paths]}
        class_counts = {}  # {label: num_images}

        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path) and class_name in self.label_map:
                label = self.label_map[class_name]
                img_files = [
                    os.path.join(class_path, img_file)
                    for img_file in os.listdir(class_path) 
                    if os.path.splitext(img_file)[1].lower() in allowed_extensions]
                class_images[label] = img_files
                class_counts[label] = len(img_files)

        # Perform basic augmentation for each class, doubling the data for each class
        augmented_images = {label: [] for label in class_images.keys()}
        
        for label, img_list in class_images.items():
            self.images.extend(img_list)
            self.labels.extend([label] * len(img_list))

            # Augment each image once
            augmented_images[label].extend([self._augment_image(img_path) for img_path in img_list])
            self.images.extend(augmented_images[label])
            self.labels.extend([label] * len(augmented_images[label]))

        # Get the doubled class count
        max_count = max(len(self.images) // 2, len(augmented_images[0]))

        # Balance the classes
        for label, img_list in class_images.items():
            current_count = class_counts[label] * 2  # Includes count after one augmentation

            # If class samples are less than the maximum count, add more augmented versions to balance
            if current_count < max_count:
                additional_images_needed = max_count - current_count
                for i in range(additional_images_needed):
                    self.images.append(augmented_images[label][i % len(augmented_images[label])])
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load and preprocess image
        image = Image.open(img_path.replace("_aug", "")).convert("L")

        # Determine if augmentation is needed
        if self.phase == 'train' and self.transform and "_aug" in img_path:
            image = self.transform(image)  # Apply data augmentation
        else:
            basic_transform = transforms.Compose([
                transforms.Resize((1200, 1200)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            image = basic_transform(image)
                
        return image, label

    def _augment_image(self, img_path):
        """Generate a copy of the augmented image path, marked as an augmented version."""
        return f"{img_path}_aug"

# Define common data augmentation and preprocessing functions
def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((1200,1200)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  
        ])
    else:
        # For 'validation' and 'test' phases, use the same preprocessing
        return transforms.Compose([
            transforms.Resize((1200, 1200)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

def create_data_loaders_ddp(data_dir, batch_size=32, num_workers=4):
    # Create dataset objects
    train_data = CustomXrayDataset(data_dir=os.path.join(data_dir, 'train'),
                                   phase='train',
                                   transform=get_transforms('train'))
    val_data = CustomXrayDataset(data_dir=os.path.join(data_dir, 'validation'),
                                 phase='validation',
                                 transform=get_transforms('validation'))
    test_data = CustomXrayDataset(data_dir=os.path.join(data_dir, 'test'),
                                  phase='test',
                                  transform=get_transforms('test'))

    # Use DistributedSampler to distribute data across different processes
    train_sampler = DistributedSampler(train_data)
    val_sampler = DistributedSampler(val_data)
    test_sampler = DistributedSampler(test_data)

    # Create DataLoader with sampler parameter
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
