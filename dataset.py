import os
import math
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms

class GrayToRGB:
    """将单通道灰度图像转换为三通道RGB图像的转换器。"""
    def __call__(self, image):
        return image.convert('RGB')

class CustomXrayDataset(Dataset):
    def __init__(self, data_dir, phase='train', expected_size=(1200, 1200)):
        self.data_dir = data_dir
        self.phase = phase.lower()
        self.expected_size = expected_size

        # 类别名称到数字的映射
        self.label_map = {"equiax": 0, "columnar": 1}
        self.images = []
        self.labels = []
        self.image_transforms = []

        allowed_extensions = {".jpeg", ".jpg", ".png"}
        class_images = {}

        # 收集图像路径和对应的标签
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path) and class_name in self.label_map:
                label = self.label_map[class_name]
                img_files = [
                    os.path.join(class_path, img_file)
                    for img_file in os.listdir(class_path)
                    if os.path.splitext(img_file)[1].lower() in allowed_extensions]
                class_images[label] = img_files

        # 计算每个类别的最大样本数量
        max_samples_per_class = 2 * max(len(class_images[label]) for label in class_images)

        # 初始化平衡后的列表
        balanced_images = []
        balanced_labels = []
        balanced_transforms = []

        for label, img_list in class_images.items():
            num_images = len(img_list)
            # 计算需要重复的次数
            num_duplicates = math.ceil(max_samples_per_class / (2 * num_images))
            total_samples = 0
            for img_path in img_list:
                for _ in range(num_duplicates):
                    # 添加原始图像
                    balanced_images.append(img_path)
                    balanced_labels.append(label)
                    balanced_transforms.append('original')

                    # 添加增强后的图像
                    balanced_images.append(img_path)
                    balanced_labels.append(label)
                    balanced_transforms.append('augmented')
                    total_samples += 2
                    if total_samples >= max_samples_per_class:
                        break
                if total_samples >= max_samples_per_class:
                    break

        # 将平衡后的列表赋值给数据集
        self.images = balanced_images
        self.labels = balanced_labels
        self.image_transforms = balanced_transforms

        # 定义转换
        self.basic_transform = transforms.Compose([
            GrayToRGB(),
            transforms.Resize(self.expected_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.augmentation_transform = transforms.Compose([
            GrayToRGB(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomChoice([
                transforms.RandomRotation(degrees=[0, 0]),
                transforms.RandomRotation(degrees=[90, 90]),
                transforms.RandomRotation(degrees=[180, 180]),
                transforms.RandomRotation(degrees=[270, 270])
            ]),
            transforms.RandomCrop(self.expected_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        transform_type = self.image_transforms[idx]

        # 加载图像
        image = Image.open(img_path)

        # 根据transform_type应用相应的转换
        if transform_type == 'original':
            # 应用基本转换（不进行数据增强）
            image = self.basic_transform(image)
        elif transform_type == 'augmented':
            # 应用数据增强转换
            if self.phase == 'train':
                image = self.augmentation_transform(image)
            else:
                image = self.basic_transform(image)

        return image, label

def create_data_loaders_ddp(data_dir, batch_size=32, num_workers=25):
    # 创建数据集对象
    train_data = CustomXrayDataset(data_dir=os.path.join(data_dir, 'train'),
                                   phase='train')
    val_data = CustomXrayDataset(data_dir=os.path.join(data_dir, 'validation'),
                                 phase='validation')
    test_data = CustomXrayDataset(data_dir=os.path.join(data_dir, 'test'),
                                  phase='test')

    # 使用 DistributedSampler 进行数据分配
    train_sampler = DistributedSampler(train_data)
    val_sampler = DistributedSampler(val_data)
    test_sampler = DistributedSampler(test_data)

    # 创建 DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
