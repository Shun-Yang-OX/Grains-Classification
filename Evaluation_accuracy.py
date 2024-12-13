import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import Model
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"正在使用的设备: {device}")

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, loader=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.loader = loader if loader is not None else self.default_loader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.loader(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label, image_path

    @staticmethod
    def default_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

def main(data_dir, result_folder, model_weights_path, num_classes, batch_size, seed):
    utils.set_seed(seed)
    model = Model.build_resnet152_for_xray(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_weights_path, map_location=device)

    # 去除'module.'前缀
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((1200, 1200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(data_dir, 'test')
    categories = ['columnar', 'equiax']
    class_to_label = {"equiax": 0, "columnar": 1}

    for category in categories:
        category_path = os.path.join(test_dir, category)
        label = class_to_label[category]

        subfolders = [name for name in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, name))]
        for subfolder in subfolders:
            subfolder_path = os.path.join(category_path, subfolder)
            experiment_result_folder = os.path.join(result_folder, f"{category}_{subfolder}")
            os.makedirs(experiment_result_folder, exist_ok=True)

            image_paths = []
            for root, _, files in os.walk(subfolder_path):
                # 为了保证顺序，在添加文件名时进行排序
                # 可以先对files排序，然后加入image_paths
                files = sorted(files)
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        image_paths.append(os.path.join(root, file))

            if len(image_paths) == 0:
                print(f"{subfolder_path} 中未找到图像。跳过...")
                continue

            # 对image_paths排序，以确保加载顺序稳定
            image_paths.sort()

            labels = [label] * len(image_paths)
            test_dataset = CustomImageDataset(image_paths, labels, transform=data_transforms)
            data_loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            y_true, y_pred = [], []
            image_results = []
            image_probabilities_pos = []  # 每张图片对正类(columnar)的概率
            image_probabilities_neg = []  # 每张图片对负类(equiax)的概率

            with torch.no_grad():
                for images, labels_batch, paths_batch in tqdm(data_loader_test, desc='Evaluating'):
                    images = images.to(device)
                    outputs = model(images)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted = torch.argmax(probabilities, dim=1)

                    y_true.extend(labels_batch.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

                    for i in range(len(labels_batch)):
                        true_label = labels_batch[i].item()
                        pred_label = predicted[i].item()
                        file_path = paths_batch[i]

                        pos_prob = probabilities[i, 1].item()
                        neg_prob = probabilities[i, 0].item()

                        image_results.append({
                            'Filename': file_path,
                            'True_Label': true_label,
                            'Predicted_Label': pred_label,
                            'Pos_Prob': pos_prob,
                            'Neg_Prob': neg_prob
                        })

                        image_probabilities_pos.append(pos_prob)
                        image_probabilities_neg.append(neg_prob)

            # 保存分类结果
            subfolder_results_df = pd.DataFrame(image_results)
            subfolder_csv_path = os.path.join(experiment_result_folder, "image_results.csv")
            subfolder_results_df.to_csv(subfolder_csv_path, index=False)
            print(f"分类结果已保存到 {subfolder_csv_path}")

            # 绘制混淆矩阵
            conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            conf_matrix_path = os.path.join(experiment_result_folder, "confusion_matrix.png")
            plt.savefig(conf_matrix_path)
            plt.close()
            print(f"混淆矩阵已保存到 {conf_matrix_path}")

            # 根据概率动态调整Y轴范围，让趋势更明显
            def set_y_axis(prob_array):
                p_min = min(prob_array)
                p_max = max(prob_array)
                if p_min == p_max:  # 所有值都相同的情况
                    return p_min - 0.01, p_max + 0.01
                else:
                    # 给上下边界留点余量
                    margin = 0.05 * (p_max - p_min)
                    lower = p_min - margin
                    upper = p_max + margin
                    # 保证范围在[0,1]之间
                    return max(0, lower), min(1, upper)

            x_indices = range(len(image_probabilities_pos))

            # 正类概率曲线
            pos_ymin, pos_ymax = set_y_axis(image_probabilities_pos)
            plt.figure(figsize=(12, 6))
            plt.plot(x_indices, image_probabilities_pos, marker='o', linestyle='-', color='blue')
            plt.ylim(pos_ymin, pos_ymax)
            plt.xlabel('Step (Image Index)', fontsize=14)
            plt.ylabel('Positive Probability', fontsize=14)
            plt.title('Positive Probability per Image', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_result_folder, "positive_probability_per_image.png"))
            plt.close()

            # 负类概率曲线
            neg_ymin, neg_ymax = set_y_axis(image_probabilities_neg)
            plt.figure(figsize=(12, 6))
            plt.plot(x_indices, image_probabilities_neg, marker='o', linestyle='-', color='red')
            plt.ylim(neg_ymin, neg_ymax)
            plt.xlabel('Step (Image Index)', fontsize=14)
            plt.ylabel('Negative Probability', fontsize=14)
            plt.title('Negative Probability per Image', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_result_folder, "negative_probability_per_image.png"))
            plt.close()

            print(f"概率曲线已保存到 {experiment_result_folder}")


if __name__ == "__main__":
    DATA_DIR = r'/home/shun/Project/Grains-Classification/Dataset/Test_val'
    RESULT_FOLDER = r'/home/shun/Project/Grains-Classification/Result_test/ResNet_unfrozen_test_val'
    MODEL_WEIGHTS_PATH = r'/home/shun/Project/Grains-Classification/Result_test/ResNet_unfrozen_checkpoints/best_model_epoch_12_val_loss_0.0000.pth'
    num_classes = 2
    batch_size = 1
    seed = 10086

    main(DATA_DIR, RESULT_FOLDER, MODEL_WEIGHTS_PATH, num_classes, batch_size, seed)
