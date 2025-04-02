import os
import cv2
import numpy as np
from PIL import Image
import torch

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class OriginalCAM:
    """
    原始 CAM 的实现：
    1. 在目标卷积层上注册 forward hook，保存该层输出的 feature maps。
    2. 利用模型 fc 层的权重，对每个样本计算 CAM（加权求和后经过 ReLU 和归一化）。
    
    注意：该方法仅适用于具有全局平均池化层和 fc 层（如 ResNet）的模型。
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.hook = self.target_layer.register_forward_hook(self._hook_fn) 
    
    def _hook_fn(self, module, input, output):
        self.feature_maps = output.detach()
    
    def compute_cam(self, input_tensor, targets=None):
        # 运行一次 forward pass，hook 会捕获 feature maps
        outputs = self.model(input_tensor)
        # 如果没有指定目标类别，则使用预测结果
        if targets is None:
            targets = outputs.argmax(dim=1)
            
        # 确保模型有fc属性 
        if not hasattr(self.model, 'fc'):
            raise AttributeError("模型没有'fc'属性，OriginalCAM仅适用于带有fc层的模型（如ResNet）")
            
        # 获取fc层权重
        fc_weights = self.model.fc.weight.detach()  # shape: [num_classes, channels]
        
        # 确保feature_maps非空
        if self.feature_maps is None:
            raise ValueError("feature_maps为空，请确保在调用compute_cam前已经执行了forward pass")
            
        B, C, H, W = self.feature_maps.shape
        cams = []
        
        for i in range(B):
            # 直接从ClassifierOutputTarget中提取类别索引
            try:
                # 首先尝试直接获取target_category属性
                if hasattr(targets[i], 'target_category'):
                    class_idx = targets[i].target_category
                else:
                    # 如果没有target_category属性，则尝试将targets[i]作为索引
                    class_idx = targets[i]
                
                # 确保class_idx是整数
                if hasattr(class_idx, 'item'):
                    # 如果class_idx是张量，调用item()方法
                    class_idx = class_idx.item()
                else:
                    # 否则直接转换为整数
                    class_idx = int(class_idx)
                    
            except Exception as e:
                print(f"处理类别索引时出错: {e}")
                print(f"targets[{i}]的类型: {type(targets[i])}")
                if hasattr(targets[i], 'target_category'):
                    print(f"target_category的类型: {type(targets[i].target_category)}")
                # 使用默认值作为后备
                class_idx = 0
            
            # 获取对应类别的权重
            weights = fc_weights[class_idx]  # original shape of weights:[channels, weights].   fc_weights[class_idx] certain class'weight.       
            # 计算加权和
            cam = (weights.view(1, C, 1, 1) * self.feature_maps[i:i+1]).sum(dim=1)
            cam = torch.relu(cam)
            
            # 归一化 CAM
            cam_min = cam.min()
            cam_max = cam.max()
            
            if cam_max != cam_min:  # 避免除以零
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)
                
            cams.append(cam.squeeze(0).cpu().numpy())
            
        return cams
        
    def __call__(self, input_tensor, targets=None):
        """
        使原始CAM对象可调用，与pytorch_grad_cam一致的接口
        """
        try:
            return self.compute_cam(input_tensor, targets)
        except Exception as e:
            print(f"OriginalCAM计算错误: {e}")
            # 返回一个合适的默认值，例如全零热力图
            B = input_tensor.shape[0]
            # 假设特征图的空间维度是正方形
            if self.feature_maps is not None:
                H, W = self.feature_maps.shape[2], self.feature_maps.shape[3]
                return [np.zeros((H, W)) for _ in range(B)]
            else:
                # 如果没有特征图，返回输入尺寸的热力图
                return [np.zeros((input_tensor.shape[2], input_tensor.shape[3])) for _ in range(B)]

def build_cam(model, model_type, cam_method='gradcam'):
    """
    根据已构建好的模型和模型类型构建 CAM 对象。

    参数:
        model (torch.nn.Module): 已加载权重并处于评估模式的模型
        model_type (str): 模型类型，例如 'resnet', 'swin' 或 'swin_v2'
        cam_method (str): CAM 方法，支持 'gradcam', 'gradcam++', 'scorecam', 'original_cam'
    
    返回:
        cam: 根据指定方法构建的 CAM 对象
    """
    cam_method = cam_method.lower()
    if cam_method == 'original_cam':
        if model_type.lower() == 'resnet':
            # 对于原始 CAM，我们选取 ResNet 的最后卷积层
            target_layer = model.layer4[-1].conv3
            try:
                return OriginalCAM(model, target_layer)
            except Exception as e:
                print(f"创建OriginalCAM时出错: {e}")
                print("尝试使用GradCAM作为后备方法...")
                target_layers = [target_layer]
                return GradCAM(model=model, target_layers=target_layers)
        else:
            print(f"警告: 原始CAM方法不适用于{model_type}模型，将使用GradCAM代替。")
            if model_type.lower() == "resnet":
                target_layers = [model.layer4[-1].conv3]
            elif model_type.lower() in ["swin", "swin_v2"]:
                target_layers = [model.layers[-1].blocks[-1].norm2]
            else:
                raise ValueError(f"不支持的 model_type: {model_type}")
            return GradCAM(model=model, target_layers=target_layers)
    else:
        # 对于基于梯度的 CAM 方法，选择目标层
        if model_type.lower() == "resnet":
            target_layers = [model.layer4[-1].conv3]
        elif model_type.lower() in ["swin", "swin_v2"]:
            target_layers = [model.layers[-1].blocks[-1].norm2]
        else:
            raise ValueError(f"不支持的 model_type: {model_type}")
        
        reshape_transform = None
        if model_type.lower() in ["swin", "swin_v2"]:
            def reshape_transform(tensor):
                import math
                B, N, C = tensor.shape
                h = w = int(math.sqrt(N))
                if h * w != N:
                    raise ValueError(f"N={N} 不能整除成正方形")
                result = tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)
                return result
            reshape_transform = reshape_transform
        
        if cam_method == 'gradcam':
            cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        elif cam_method == 'gradcam++':
            cam = GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        elif cam_method == 'scorecam':
            cam = ScoreCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        else:
            raise ValueError(f"不支持的 cam_method: {cam_method}")
        return cam

def generate_cam_heatmaps(images, predicted, paths, experiment_result_folder, cam, target_size=(1056, 1056)):
    """
    生成并保存 CAM 热力图。

    参数:
        images: 输入模型的图像张量（已移至 device）
        predicted: 模型预测的类别（tensor）
        paths: 图像文件路径列表
        experiment_result_folder (str): 保存 CAM 热力图的目录
        cam: 通过 build_cam 得到的 CAM 对象
        target_size (tuple): 生成热力图时调整的图像尺寸
    """
    # 构造每个样本的目标
    try:
        # 转换预测结果为int，避免张量问题
        targets = []
        for pred in predicted:
            if hasattr(pred, 'item'):
                class_idx = pred.item()
            else:
                class_idx = int(pred)
            targets.append(class_idx)
    except Exception as e:
        print(f"创建targets时出错: {e}")
        # 作为备用，直接使用预测的类别索引
        targets = predicted.cpu().tolist()
    
    try:
        # 对 CAM 对象进行调用，获得灰度热力图
        grayscale_cams = cam(input_tensor=images, targets=targets)
    except Exception as e:
        print(f"CAM计算错误: {e}")
        print(f"CAM对象类型: {type(cam)}")
        print("尝试使用备用方法...")
        
        try:
            # 尝试不同的调用方式
            if hasattr(cam, 'compute_cam'):
                grayscale_cams = cam.compute_cam(input_tensor=images, targets=targets)
            else:
                # 创建默认的空热力图
                grayscale_cams = [np.zeros((target_size[1], target_size[0])) for _ in range(len(paths))]
        except Exception as e2:
            print(f"备用方法也失败: {e2}")
            return
    
    for i, file_path in enumerate(paths):
        try:
            # 获取当前样本的灰度热力图
            grayscale_cam = grayscale_cams[i]
            
            # 打印热力图的形状以便调试
            print(f"热力图原始形状: {grayscale_cam.shape}")
            
            # 将热力图调整为与目标图像相同的大小
            # 确保热力图是2D的灰度图（只有高和宽，没有通道）
            if len(grayscale_cam.shape) == 2:
                # 使用cv2调整大小，确保热力图是浮点数格式
                grayscale_cam = cv2.resize(grayscale_cam, target_size, interpolation=cv2.INTER_LINEAR)
            else:
                print(f"警告: 热力图不是2D的，形状为 {grayscale_cam.shape}")
                # 尝试将其转换为2D，如果可能的话
                if len(grayscale_cam.shape) == 3 and grayscale_cam.shape[2] == 1:
                    grayscale_cam = grayscale_cam[:, :, 0]
                    grayscale_cam = cv2.resize(grayscale_cam, target_size, interpolation=cv2.INTER_LINEAR)
                else:
                    # 创建一个全零的替代热力图
                    grayscale_cam = np.zeros(target_size)
            
            # 打开原始图像并调整大小
            pil_img = Image.open(file_path).convert('RGB')
            pil_img = pil_img.resize(target_size)
            rgb_img = np.array(pil_img, dtype=np.float32) / 255.0
            
            # 确保热力图的范围在0-1之间
            if grayscale_cam.min() < 0 or grayscale_cam.max() > 1:
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
            
            # 再次打印调整后的热力图形状
            print(f"调整后的热力图形状: {grayscale_cam.shape}")
            print(f"RGB图像形状: {rgb_img.shape}")
            
            # 使用show_cam_on_image函数叠加热力图和原始图像
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            
            # 保存结果
            base_name = os.path.basename(file_path)
            fname_no_ext, ext = os.path.splitext(base_name)
            save_path = os.path.join(experiment_result_folder, f"{fname_no_ext}_cam{ext}")
            cv2.imwrite(save_path, cam_image_bgr)
            
        except Exception as e:
            print(f"处理图像 {file_path} 时出错: {e}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈
            continue