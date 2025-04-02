# Evaluation_4_classes.py

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2

# 导入 CAM 相关函数
import cam_utils
import Model  
import utils_setup  
import utils_analysis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

def run_evaluation(data_dir, result_folder, model_weights_path, num_classes, batch_size, seed, model_type, output_cam=False, cam_method='gradcam'):
    """
    运行评估流程：
      - 调用 Model.py 构建模型并加载权重
      - 根据模型构建 CAM 对象（如果 output_cam 为 True 则生成 CAM 热力图）
      - 对 test 数据集进行评估，并保存各子文件夹下的分类结果、混淆矩阵、概率曲线等
    """
    utils_setup.set_seed(seed)
    
    # 构建模型（调用 Model.py 中的 build 函数）
    if model_type.lower() == "resnet":
        model = Model.build_resnet152_for_xray(num_classes=num_classes, pretrained=True, freeze_backbone=True)
    elif model_type.lower() == "swin":
        model = Model.build_swin_transformer_model(num_classes=num_classes)
    elif model_type.lower() == "swin_v2":
        model = Model.build_swin_transformer_v2_model(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的 model_type: {model_type}")
    
    # 加载权重
    checkpoint = torch.load(model_weights_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {key.replace('module.', ''): state_dict[key] for key in state_dict}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # 若需要 CAM 可视化，则利用 cam_utils 构建 CAM 对象
    if output_cam:
        cam = cam_utils.build_cam(model, model_type, cam_method=cam_method)
    else:
        cam = None
    
    # 定义测试预处理：CenterCrop 到 1056x1056
    data_transforms = transforms.Compose([
        transforms.CenterCrop(1056),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_dir = os.path.join(data_dir, 'test')
    categories = ['columnar', 'equiax', 'background', 'IMC']
    class_to_label = {"equiax": 0, "columnar": 1, "background": 2, "IMC": 3}
    overall_y_true = []
    overall_y_pred = []
    
    for category in categories:
        category_path = os.path.join(test_dir, category)
        if category not in class_to_label:
            print(f"类别 {category} 不在 label 字典中，跳过...")
            continue
        label = class_to_label[category]
        subfolders = [name for name in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, name))]
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(category_path, subfolder)
            experiment_result_folder = os.path.join(result_folder, f"{category}_{subfolder}")
            os.makedirs(experiment_result_folder, exist_ok=True)
            
            image_paths = []
            for root, _, files in os.walk(subfolder_path):
                for file in sorted(files):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        image_paths.append(os.path.join(root, file))
            
            if not image_paths:
                print(f"{subfolder_path} 中未找到图像，跳过...")
                continue
            
            image_paths.sort()
            labels_list = [label] * len(image_paths)
            test_dataset = CustomImageDataset(image_paths, labels_list, transform=data_transforms)
            data_loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            
            y_true, y_pred = [], []
            image_results = []
            image_probabilities_class0 = []
            image_probabilities_class1 = []
            image_probabilities_class2 = []
            image_probabilities_class3 = []
            
            for images, labels_batch, paths_batch in tqdm(data_loader_test, desc='Evaluating'):
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                
                # 若开启 CAM 输出，则生成热力图
                if output_cam and cam is not None:
                    cam_utils.generate_cam_heatmaps(images, predicted, paths_batch, experiment_result_folder, cam, target_size=(1056, 1056))
                
                y_true.extend(labels_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
                for i in range(len(labels_batch)):
                    prob0 = probabilities[i, 0].item()
                    prob1 = probabilities[i, 1].item()
                    prob2 = probabilities[i, 2].item()
                    prob3 = probabilities[i, 3].item()
                    image_results.append({
                        'Filename': paths_batch[i],
                        'True_Label': labels_batch[i].item(),
                        'Predicted_Label': predicted[i].item(),
                        'Prob_Class_0': prob0,
                        'Prob_Class_1': prob1,
                        'Prob_Class_2': prob2,
                        'Prob_Class_3': prob3
                    })
                    image_probabilities_class0.append(prob0)
                    image_probabilities_class1.append(prob1)
                    image_probabilities_class2.append(prob2)
                    image_probabilities_class3.append(prob3)
            
            overall_y_true.extend(y_true)
            overall_y_pred.extend(y_pred)
            
            subfolder_results_df = pd.DataFrame(image_results)
            subfolder_csv_path = os.path.join(experiment_result_folder, "image_results.csv")
            subfolder_results_df.to_csv(subfolder_csv_path, index=False)
            print(f"分类结果已保存至 {subfolder_csv_path}")
            
            conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=['Equiaxed', 'Columnar', 'Background', 'IMC'],
                        yticklabels=['Equiaxed', 'Columnar', 'Background', 'IMC'])
            plt.title('混淆矩阵')
            plt.xlabel('预测')
            plt.ylabel('真实')
            plt.tight_layout()
            conf_matrix_path = os.path.join(experiment_result_folder, "confusion_matrix.png")
            plt.savefig(conf_matrix_path)
            plt.close()
            print(f"混淆矩阵保存至 {conf_matrix_path}")
            
            x_indices = range(len(image_probabilities_class0))
            all_probs = (image_probabilities_class0 + image_probabilities_class1 +
                         image_probabilities_class2 + image_probabilities_class3)
            ymin = min(all_probs) - 0.01
            ymax = max(all_probs) + 0.01
            plt.figure(figsize=(12, 6))
            plt.plot(x_indices, image_probabilities_class0, marker='o', label='Class 0')
            plt.plot(x_indices, image_probabilities_class1, marker='^', label='Class 1')
            plt.plot(x_indices, image_probabilities_class2, marker='s', label='Class 2')
            plt.plot(x_indices, image_probabilities_class3, marker='d', label='IMC')
            plt.ylim(ymin, ymax)
            plt.xlabel('图像索引')
            plt.ylabel('概率')
            plt.title('各图像类别概率')
            plt.legend()
            plt.tight_layout()
            prob_plot_path = os.path.join(experiment_result_folder, "Probability_per_video.png")
            plt.savefig(prob_plot_path)
            plt.close()
            print(f"概率曲线图保存至 {prob_plot_path}")
    
    overall_accuracy = accuracy_score(overall_y_true, overall_y_pred)
    print(f"总体准确率: {overall_accuracy * 100:.2f}%")
    
    overall_results_df = pd.DataFrame({'Overall_Accuracy': [overall_accuracy]})
    overall_results_path = os.path.join(result_folder, "overall_results.csv")
    overall_results_df.to_csv(overall_results_path, index=False)
    print(f"总体准确率结果保存至 {overall_results_path}")
    
    overall_conf_matrix = confusion_matrix(overall_y_true, overall_y_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    sns.heatmap(overall_conf_matrix, annot=True, fmt="d", cmap="Greens",
                xticklabels=['Equiaxed', 'Columnar', 'Background', 'IMC'],
                yticklabels=['Equiaxed', 'Columnar', 'Background', 'IMC'])
    plt.title('总体混淆矩阵')
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.tight_layout()
    overall_conf_matrix_path = os.path.join(result_folder, "overall_confusion_matrix.png")
    plt.savefig(overall_conf_matrix_path)
    plt.close()
    print(f"总体混淆矩阵保存至 {overall_conf_matrix_path}")
    
    # 调用原有功能（动画生成、每视频准确率统计、整体平衡准确率计算）
    utils_analysis.generate_animations(result_folder, data_dir, with_heatmap=True)
    utils_analysis.calculate_accuracy_for_folders(result_folder)
    utils_analysis.compute_overall_balanced_accuracy(result_folder)
    
    return overall_accuracy

def main():
    DATA_DIR = r'/home/shun/Project/Grains-Classification/Dataset/Test_demo'
    RESULT_FOLDER = r'/home/shun/Project/Grains-Classification/Result_CAM/evaluation'
    MODEL_WEIGHTS_PATH = r'/home/shun/Project/Grains-Classification/Result_CAM/checkpoints/best_model_epoch_20_val_loss_0.2420.pth'
    num_classes = 4
    batch_size = 1
    seed = 10086
    model_type = "swin"  # 或者 'resnet'
    
    get_evaluation = True
    output_cam = True         # 是否生成 CAM 热力图
    cam_method = 'gradcam++'    # 可选 'gradcam', 'gradcam++', 'scorecam', 'original_cam'
    
    if get_evaluation:
        run_evaluation(DATA_DIR, RESULT_FOLDER, MODEL_WEIGHTS_PATH, num_classes, batch_size, seed, model_type, output_cam, cam_method)

if __name__ == "__main__":
    main()
