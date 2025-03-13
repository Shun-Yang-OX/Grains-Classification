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
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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

def build_model_and_cam(model_type, model_weights_path, num_classes):
    if model_type.lower() == "resnet":
        model = Model.build_resnet152_for_xray(num_classes=num_classes, pretrained=True)
    elif model_type.lower() == "swin":
        model = Model.build_swin_transformer_model(num_classes=num_classes)
    elif model_type.lower() == "swin_v2":
        model = Model.build_swin_transformer_v2_model(num_classes=num_classes)
    else:
        raise ValueError(f"model_type='{model_type}' not supported. Please use 'resnet' or 'swin'.")
    
    checkpoint = torch.load(model_weights_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {key.replace('module.', ''): state_dict[key] for key in state_dict}
    model.load_state_dict(new_state_dict)
    model.eval().to(device)
    
    if model_type.lower() == "resnet":
        target_layers = [model.layer4[-1].conv3]
        cam = GradCAM(model=model, target_layers=target_layers)
    elif model_type.lower() in ["swin", "swin_v2"]:
        target_layers = [model.layers[-1].blocks[-1].norm2]
        def reshape_transform(tensor):
            import math
            B, N, C = tensor.shape
            h = w = int(math.sqrt(N))
            if h * w != N:
                raise ValueError(f"N={N} is not a perfect square, cannot reshape to [H,W].")
            result = tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)
            return result
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    else:
        raise ValueError(f"model_type='{model_type}' not supported. Please use 'resnet' or 'swin'.")
    return model, cam

def generate_gradcam_heatmaps(images, predicted, paths, experiment_result_folder, cam, target_size=(1056, 1056)):
    """
    Generate and save Grad-CAM heatmaps.
    
    Parameters:
        images: The image tensor input to the model (already moved to device).
        predicted: The predicted classes from the model (tensor).
        paths: List of file paths corresponding to the images.
        experiment_result_folder: Folder path to save the results.
        cam: GradCAM object.
        target_size: The size of the image when generating the heatmap (default: 1056x1056).
    """
    # Build targets for each sample
    targets = [ClassifierOutputTarget(int(pred.cpu().numpy())) for pred in predicted]
    # Compute grad-cam
    grayscale_cams = cam(input_tensor=images, targets=targets)
    # For each image in the batch, generate and save the heatmap
    for i, file_path in enumerate(paths):
        grayscale_cam = grayscale_cams[i]
        pil_img = Image.open(file_path).convert('RGB')
        pil_img = pil_img.resize(target_size)
        rgb_img = np.array(pil_img, dtype=np.float32) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        base_name = os.path.basename(file_path)
        fname_no_ext, ext = os.path.splitext(base_name)
        save_path = os.path.join(experiment_result_folder, f"{fname_no_ext}_gradcam{ext}")
        cv2.imwrite(save_path, cam_image_bgr)

def run_evaluation(data_dir, result_folder, model_weights_path, num_classes, batch_size, seed, model_type, output_gradcam=False):
    """
    Run the evaluation process on the test set and optionally output Grad-CAM heatmaps based on the value of output_gradcam.
    """
    utils_setup.set_seed(seed)
    model, cam = build_model_and_cam(model_type, model_weights_path, num_classes)
    
    # Modify the preprocessing: directly CenterCrop to 1056x1056
    data_transforms = transforms.Compose([
        transforms.CenterCrop(1056),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_dir = os.path.join(data_dir, 'test')
    # Added "IMC" class in the category list
    categories = ['columnar', 'equiax', 'background', 'IMC']
    class_to_label = {
        "equiax": 0, 
        "columnar": 1, 
        "background": 2,
        "IMC": 3
    }
    overall_y_true = []
    overall_y_pred = []
    
    for category in categories:
        category_path = os.path.join(test_dir, category)
        if category not in class_to_label:
            print(f"Category {category} not found in class_to_label dictionary. Skipping...")
            continue
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
            image_probabilities_class3 = []  # Added
            
            for images, labels_batch, paths_batch in tqdm(data_loader_test, desc='Evaluating'):
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                
                # Generate Grad-CAM heatmaps based on the flag
                if output_gradcam:
                    generate_gradcam_heatmaps(images, predicted, paths_batch, experiment_result_folder, cam, target_size=(1056, 1056))
                
                y_true.extend(labels_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
                for i in range(len(labels_batch)):
                    prob_class_0 = probabilities[i, 0].item()
                    prob_class_1 = probabilities[i, 1].item()
                    prob_class_2 = probabilities[i, 2].item()
                    prob_class_3 = probabilities[i, 3].item()  # Added
                    
                    image_results.append({
                        'Filename': paths_batch[i],
                        'True_Label': labels_batch[i].item(),
                        'Predicted_Label': predicted[i].item(),
                        'Prob_Class_0': prob_class_0,
                        'Prob_Class_1': prob_class_1,
                        'Prob_Class_2': prob_class_2,
                        'Prob_Class_3': prob_class_3  # Added
                    })
                    image_probabilities_class0.append(prob_class_0)
                    image_probabilities_class1.append(prob_class_1)
                    image_probabilities_class2.append(prob_class_2)
                    image_probabilities_class3.append(prob_class_3)  # Added
            
            overall_y_true.extend(y_true)
            overall_y_pred.extend(y_pred)
            
            subfolder_results_df = pd.DataFrame(image_results)
            subfolder_csv_path = os.path.join(experiment_result_folder, "image_results.csv")
            subfolder_results_df.to_csv(subfolder_csv_path, index=False)
            print(f"Classification results saved to {subfolder_csv_path}")
            
            conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=['Equiaxed', 'Columnar', 'Background', 'IMC'],
                        yticklabels=['Equiaxed', 'Columnar', 'Background', 'IMC'])
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
            all_probabilities = (image_probabilities_class0 + image_probabilities_class1 +
                                 image_probabilities_class2 + image_probabilities_class3)
            ymin, ymax = set_y_axis(all_probabilities)
            plt.figure(figsize=(12, 6))
            
            plt.plot(x_indices, image_probabilities_class0, marker='o', linestyle='-', color='blue', label='Class 0')
            plt.plot(x_indices, image_probabilities_class1, marker='^', linestyle='-', color='red', label='Class 1')
            plt.plot(x_indices, image_probabilities_class2, marker='s', linestyle='-', color='green', label='Class 2')
            plt.plot(x_indices, image_probabilities_class3, marker='d', linestyle='-', color='orange', label='IMC')
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
    
    overall_conf_matrix = confusion_matrix(overall_y_true, overall_y_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    sns.heatmap(overall_conf_matrix, annot=True, fmt="d", cmap="Greens",
                xticklabels=['Equiaxed', 'Columnar', 'Background', 'IMC'],
                yticklabels=['Equiaxed', 'Columnar', 'Background', 'IMC'])
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
        output_gradcam=False,
        animation=False, 
        get_accuracy_per_video=False, 
        get_overall_balanced_accuracy=False,
        with_heatmap=True
        ):
    # Call the evaluation process
    if get_evaluation:
        run_evaluation(DATA_DIR, RESULT_FOLDER, MODEL_WEIGHTS_PATH, num_classes, batch_size, seed, model_type, output_gradcam)
    
    if animation:
        utils_analysis.generate_animations(RESULT_FOLDER, DATA_DIR, with_heatmap)
    
    if get_accuracy_per_video:
        utils_analysis.calculate_accuracy_for_folders(RESULT_FOLDER)
    
    if get_overall_balanced_accuracy:
        utils_analysis.compute_overall_balanced_accuracy(RESULT_FOLDER)

if __name__ == "__main__":
    # Parameter settings
    DATA_DIR = r'/home/shun/Project/Grains-Classification/Dataset/Test_demo'
    RESULT_FOLDER = r'/home/shun/Project/Grains-Classification/Result_2025.2.18/ST_4_class_frozen_V3/evaluation3'
    MODEL_WEIGHTS_PATH = r'/home/shun/Project/Grains-Classification/Result_2025.2.18/ST_4_class_frozen_V3/checkpoints/best_model_epoch_20_val_loss_0.2420.pth'
    num_classes = 4
    batch_size = 1
    seed = 10086
    model_type = "swin"  # Options: "resnet" or "swin"
    
    # Run evaluation
    get_evaluation = False
    # Whether to generate Grad-CAM heatmaps, True to generate, False to skip
    output_gradcam = False
    
    # Animation
    animation = True
    with_heatmap = True
    
    # Accuracy per video
    get_accuracy_per_video = False
    
    # Overall balanced accuracy
    get_overall_balanced_accuracy = False
    
    main(DATA_DIR, RESULT_FOLDER, MODEL_WEIGHTS_PATH, num_classes, batch_size, seed, model_type, 
         get_evaluation, output_gradcam, animation, get_accuracy_per_video, get_overall_balanced_accuracy, with_heatmap)
