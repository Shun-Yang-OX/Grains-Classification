import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import cv2
# 导入更多CAM方法
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

# 实现原始CAM方法
class OriginalCAM:
    """
    实现原始的Class Activation Mapping方法
    Zhou et al., 2016, "Learning Deep Features for Discriminative Localization"
    """
    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(model, target_layers, reshape_transform)
        
        # 获取最后一个全连接层的权重
        if hasattr(model, 'fc'):
            self.fc_weights = model.fc.weight.data  # for ResNet
        elif hasattr(model, 'head'):
            self.fc_weights = model.head.weight.data  # for Swin Transformer
        else:
            raise ValueError("无法找到模型的分类器权重，原始CAM可能无法正常工作")
    
    def forward(self, input_tensor):
        return self.model(input_tensor)
    
    def get_cam_weights(self, output_class_idx):
        return self.fc_weights[output_class_idx].unsqueeze(0)
    
    def __call__(self, input_tensor, targets=None):
        # 获取特征图激活
        outputs = self.activations_and_grads(input_tensor)
        
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]
        
        # 获取特征激活
        activations = self.activations_and_grads.activations[-1]
        if self.reshape_transform is not None:
            activations = self.reshape_transform(activations)
        
        cams = []
        for batch_idx, (target, activation) in enumerate(zip(targets, activations)):
            target_category = target.category
            
            # 获取分类器权重
            cam_weights = self.get_cam_weights(target_category).cpu()
            
            # 计算CAM
            weighted_activations = cam_weights.view(*cam_weights.shape, 1, 1) * activation.cpu()
            cam = weighted_activations.sum(dim=0)
            
            # 后处理CAM
            cam = np.maximum(cam.detach().numpy(), 0)  # ReLU
            
            # 归一化
            if np.max(cam) > 0:
                cam = cam / np.max(cam)
            
            # 调整大小以匹配输入大小
            cams.append(cam)
        
        return np.array(cams)

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

def build_model_and_cam(model_type, model_weights_path, num_classes, heatmap_method="gradcam"):
    """构建模型并初始化指定的CAM方法"""
    if model_type.lower() == "resnet":
        model = Model.build_resnet152_for_xray(num_classes=num_classes, pretrained=True)
    elif model_type.lower() in ["swin"]:
        model = Model.build_swin_transformer_model(num_classes=num_classes)
    else:
        raise ValueError(f"model_type='{model_type}' not supported. Please use 'resnet', 'swin' or 'swin_v2'.")

    checkpoint = torch.load(model_weights_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {key.replace('module.', ''): state_dict[key] for key in state_dict}
    model.load_state_dict(new_state_dict)
    model.eval().to(device)

    # 设置目标层
    if model_type.lower() == "resnet":
        target_layers = [model.layer4[-1].conv3]
    elif model_type.lower() in ["swin"]:
        target_layers = [model.layers[-1].blocks[-1].norm2]
        def reshape_transform(tensor):
            import math
            B, N, C = tensor.shape
            h = w = int(math.sqrt(N))
            if h * w != N:
                raise ValueError(f"N={N} is not a perfect square, cannot reshape to [H,W].")
            result = tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)
            return result
    
    # 根据指定的方法初始化CAM
    cam = None
    if heatmap_method.lower() == "gradcam":
        if model_type.lower() in ["swin"]:
            cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        else:
            cam = GradCAM(model=model, target_layers=target_layers)
    elif heatmap_method.lower() == "originalcam" or heatmap_method.lower() == "cam":
        # 原始CAM方法
        if model_type.lower() in ["swin"]:
            cam = OriginalCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        else:
            cam = OriginalCAM(model=model, target_layers=target_layers)
        print(f"Using Original CAM method with {model_type} model")
    elif heatmap_method.lower() == "eigencam":
        # 使用EigenCAM
        if model_type.lower() in ["swin"]:
            cam = EigenCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        else:
            cam = EigenCAM(model=model, target_layers=target_layers)
    elif heatmap_method.lower() == "scorecam":
        if model_type.lower() in ["swin"]:
            cam = ScoreCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        else:
            cam = ScoreCAM(model=model, target_layers=target_layers)
    
    return model, cam

def generate_heatmaps(images, predicted, paths, experiment_result_folder, cam, method_name, target_size=(1056, 1056)):
    """
    生成并保存热力图。
    
    Parameters:
        images: 输入图像张量
        predicted: 模型预测的类别
        paths: 对应图像的文件路径
        experiment_result_folder: 保存结果的文件夹
        cam: CAM对象
        method_name: CAM方法名称，用于文件命名
        target_size: 生成热力图的尺寸
    """
    targets = [ClassifierOutputTarget(int(pred.cpu().numpy())) for pred in predicted]
    
    try:
        grayscale_cams = cam(input_tensor=images, targets=targets)
    except Exception as e:
        print(f"获取热力图失败: {e}")
        return
    
    for i, file_path in enumerate(paths):
        try:
            if i >= len(grayscale_cams):
                print(f"索引 {i} 超出热力图范围 {len(grayscale_cams)}")
                continue
                
            grayscale_cam = grayscale_cams[i]
            
            # 打印调试信息
            print(f"热力图形状: {grayscale_cam.shape}, 类型: {grayscale_cam.dtype}, 范围: [{grayscale_cam.min()}, {grayscale_cam.max()}]")
            
            # 确保热力图是2D的
            if grayscale_cam.ndim > 2:
                print(f"压缩热力图从形状 {grayscale_cam.shape}")
                grayscale_cam = grayscale_cam.squeeze()
                print(f"  到形状 {grayscale_cam.shape}")
            
            # 如果热力图仍然不是2D，取第一个通道
            if grayscale_cam.ndim > 2:
                print(f"热力图仍然是多维的，取第一个通道")
                grayscale_cam = grayscale_cam[0]
            
            # 处理原始图像
            orig_img = cv2.imread(file_path)
            if orig_img is None:
                print(f"无法读取图像: {file_path}，尝试使用PIL")
                pil_img = Image.open(file_path).convert('RGB')
                orig_img = np.array(pil_img)
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            
            # 调整原始图像的大小
            orig_img = cv2.resize(orig_img, target_size)
            
            # 创建一个自定义的热力图可视化
            # 1. 归一化热力图到0-1
            if grayscale_cam.max() > 0:
                normalized_cam = grayscale_cam / grayscale_cam.max()
            else:
                normalized_cam = grayscale_cam
            
            # 2. 转换为8位单通道图像 (0-255)
            heatmap_img = np.uint8(255 * normalized_cam)
            
            # 确保heatmap_img形状正确且类型为CV_8UC1
            print(f"热力图图像形状: {heatmap_img.shape}, 类型: {heatmap_img.dtype}")
            
            # 3. 应用颜色映射
            try:
                colored_heatmap = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
                print("成功应用颜色映射")
            except Exception as e:
                print(f"应用颜色映射失败: {e}")
                print(f"正在尝试修复图像格式...")
                
                # 如果热力图不是整数类型，强制转换
                if heatmap_img.dtype != np.uint8:
                    heatmap_img = np.uint8(heatmap_img)
                
                # 如果维度不对，确保是2D的
                if heatmap_img.ndim != 2:
                    print(f"热力图维度不对: {heatmap_img.ndim}")
                    if heatmap_img.ndim == 3 and heatmap_img.shape[2] == 1:
                        heatmap_img = heatmap_img[:, :, 0]
                    else:
                        # 创建一个假热力图
                        print("创建假热力图")
                        heatmap_img = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
                
                try:
                    colored_heatmap = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
                    print("第二次尝试应用颜色映射成功")
                except Exception as e2:
                    print(f"第二次应用颜色映射仍然失败: {e2}")
                    # 如果仍然失败，创建一个假的热力图
                    colored_heatmap = np.zeros_like(orig_img)
            
            # 4. 叠加热力图和原始图像
            alpha = 0.4  # 热力图权重
            beta = 0.6   # 原始图像权重
            cam_image_bgr = cv2.addWeighted(colored_heatmap, alpha, orig_img, beta, 0)
            
            # 保存热力图
            base_name = os.path.basename(file_path)
            fname_no_ext, ext = os.path.splitext(base_name)
            save_path = os.path.join(experiment_result_folder, f"{fname_no_ext}_{method_name}{ext}")
            cv2.imwrite(save_path, cam_image_bgr)
            print(f"已保存热力图: {save_path}")
            
        except Exception as e:
            print(f"处理图像 {file_path} 时发生错误: {e}")

def run_evaluation(data_dir, result_folder, model_weights_path, num_classes, batch_size, seed, model_type, 
                   output_heatmap=False, heatmap_method="gradcam"):
    """
    对测试集运行评估过程，并可选择输出热力图。
    
    Parameters:
        output_heatmap: 是否生成热力图
        heatmap_method: 使用的CAM方法 ("gradcam", "cam", "eigencam", "scorecam" 等)
    """
    utils_setup.set_seed(seed)
    model, cam = build_model_and_cam(model_type, model_weights_path, num_classes, heatmap_method)
    
    if cam is None:
        print(f"警告: 无法初始化 {heatmap_method} 方法，热力图生成将被禁用")
        output_heatmap = False

    # Modify the preprocessing: first center crop to a larger size, then convert to tensor and normalize
    data_transforms = transforms.Compose([
        transforms.CenterCrop(1056),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(data_dir, 'test')
    # Names and labels corresponding to the three classes
    categories = ['columnar', 'equiax', 'background']
    class_to_label = {
        "equiax": 0, 
        "columnar": 1, 
        "background": 2
    }
    overall_y_true = []
    overall_y_pred = []

    for category in categories:
        category_path = os.path.join(test_dir, category)
        label = class_to_label[category]
        subfolders = [name for name in os.listdir(category_path)
                      if os.path.isdir(os.path.join(category_path, name))]

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
                print(f"No images found in {subfolder_path}. Skipping...")
                continue

            image_paths.sort()
            labels = [label] * len(image_paths)
            test_dataset = CustomImageDataset(image_paths, labels, transform=data_transforms)
            data_loader_test = DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=4)

            y_true, y_pred = [], []
            image_results = []
            image_probabilities_class0 = []
            image_probabilities_class1 = []
            image_probabilities_class2 = []

            for images, labels_batch, paths_batch in tqdm(data_loader_test, desc='Evaluating'):
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)

                # 根据配置生成热力图
                if output_heatmap and cam is not None:
                    try:
                        generate_heatmaps(images, predicted, paths_batch, experiment_result_folder, cam, heatmap_method, target_size=(1056,1056))
                    except Exception as e:
                        print(f"生成热力图时发生错误: {e}")

                y_true.extend(labels_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                for i in range(len(labels_batch)):
                    prob_class_0 = probabilities[i, 0].item()
                    prob_class_1 = probabilities[i, 1].item()
                    prob_class_2 = probabilities[i, 2].item()

                    image_results.append({
                        'Filename': paths_batch[i],
                        'True_Label': labels_batch[i].item(),
                        'Predicted_Label': predicted[i].item(),
                        'Prob_Class_0': prob_class_0,
                        'Prob_Class_1': prob_class_1,
                        'Prob_Class_2': prob_class_2
                    })
                    image_probabilities_class0.append(prob_class_0)
                    image_probabilities_class1.append(prob_class_1)
                    image_probabilities_class2.append(prob_class_2)

            overall_y_true.extend(y_true)
            overall_y_pred.extend(y_pred)

            subfolder_results_df = pd.DataFrame(image_results)
            subfolder_csv_path = os.path.join(experiment_result_folder, "image_results.csv")
            subfolder_results_df.to_csv(subfolder_csv_path, index=False)
            print(f"Classification results saved to {subfolder_csv_path}")

            conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=['Equiaxed', 'Columnar', 'Background'],
                        yticklabels=['Equiaxed', 'Columnar', 'Background'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            conf_matrix_path = os.path.join(experiment_result_folder, "confusion_matrix.png")
            plt.savefig(conf_matrix_path)
            plt.close()
            print(f"Confusion matrix saved to {conf_matrix_path}")

            def set_y_axis(prob_array):
                p_min = min(prob_array)
                p_max = max(prob_array)
                if p_min == p_max:
                    return p_min - 0.01, p_max + 0.01
                margin = 0.05 * (p_max - p_min)
                lower = p_min - margin
                upper = p_max + margin
                return max(0, lower), min(1, upper)

            x_indices = range(len(image_probabilities_class0))
            all_probabilities = image_probabilities_class0 + image_probabilities_class1 + image_probabilities_class2
            ymin, ymax = set_y_axis(all_probabilities)
            plt.figure(figsize=(12, 6))
            plt.plot(x_indices, image_probabilities_class0, marker='o', linestyle='-', color='blue', label='Class 0')
            plt.plot(x_indices, image_probabilities_class1, marker='^', linestyle='-', color='red', label='Class 1')
            plt.plot(x_indices, image_probabilities_class2, marker='s', linestyle='-', color='green', label='Class 2')
            plt.ylim(ymin, ymax)
            plt.xlabel('Image Index', fontsize=14)
            plt.ylabel('Probability', fontsize=14)
            plt.title('Probability per Image for Each Class', fontsize=16)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_result_folder, "Probability_per_video.png"))
            plt.close()
            print(f"Probability curves saved to {experiment_result_folder}")

    overall_accuracy = accuracy_score(overall_y_true, overall_y_pred)
    print(f"Overall model accuracy: {overall_accuracy * 100:.2f}%")

    overall_results_path = os.path.join(result_folder, "overall_results.csv")
    overall_results_df = pd.DataFrame({'Overall_Accuracy': [overall_accuracy]})
    overall_results_df.to_csv(overall_results_path, index=False)
    print(f"Overall accuracy saved to {overall_results_path}")

    overall_conf_matrix = confusion_matrix(overall_y_true, overall_y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(overall_conf_matrix, annot=True, fmt="d", cmap="Greens",
                xticklabels=['Equiaxed', 'Columnar', 'Background'],
                yticklabels=['Equiaxed', 'Columnar', 'Background'])
    plt.title('Overall Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    overall_conf_matrix_path = os.path.join(result_folder, "overall_confusion_matrix.png")
    plt.savefig(overall_conf_matrix_path)
    plt.close()
    print(f"Overall confusion matrix saved to {overall_conf_matrix_path}")

def main(
        DATA_DIR, 
        RESULT_FOLDER, 
        MODEL_WEIGHTS_PATH, 
        num_classes, 
        batch_size, 
        seed, 
        model_type, 
        get_evaluation=False,
        output_heatmap=False,  # 控制是否输出热力图
        heatmap_method="gradcam",  # 指定热力图方法
        animation=False, 
        get_accuracy_per_video=False, 
        get_overall_balanced_accuracy=False,
        with_heatmap=True
        ):
    if get_evaluation:
        run_evaluation(DATA_DIR, RESULT_FOLDER, MODEL_WEIGHTS_PATH, num_classes, batch_size, seed, model_type, output_heatmap, heatmap_method)

    if animation:
        utils_analysis.generate_animations(RESULT_FOLDER, DATA_DIR, with_heatmap)
    
    if get_accuracy_per_video:
        utils_analysis.calculate_accuracy_for_folders(RESULT_FOLDER)
    
    if get_overall_balanced_accuracy:
        utils_analysis.compute_overall_balanced_accuracy(RESULT_FOLDER)

if __name__ == "__main__":
    DATA_DIR = r'/home/shun/Project/Grains-Classification/Dataset/test_val'
    RESULT_FOLDER = r'/home/shun/Project/Grains-Classification/Result_2025.2.22/ResNet_3_class_frozen/evaluation_3'
    MODEL_WEIGHTS_PATH = r'/home/shun/Project/Grains-Classification/Result_2025.2.22/ResNet_3_class_frozen/checkpoints/best_model_epoch_30_val_loss_0.0939.pth'
    num_classes = 3
    batch_size = 1
    seed = 10086
    model_type = "resnet"  # Options: "resnet", "swin" or "swin_v2"

    get_evaluation = True
    output_heatmap = True
    # 热力图生成方法选项:
    # - "gradcam": 使用梯度加权的类激活映射
    # - "cam" 或 "originalcam": 使用原始的类激活映射方法 (Zhou et al., 2016)
    # - "eigencam": 使用特征图的主成分分析 (不需要梯度)
    # - "scorecam": 使用前向传播的特征通道权重 (不需要梯度，但计算较慢)
    heatmap_method = "cam"

    animation = True
    with_heatmap = True

    get_accuracy_per_video = False
    get_overall_balanced_accuracy = False

    main(DATA_DIR, RESULT_FOLDER, MODEL_WEIGHTS_PATH, num_classes, batch_size, seed, model_type, 
         get_evaluation, output_heatmap, heatmap_method, animation, get_accuracy_per_video, get_overall_balanced_accuracy, with_heatmap)