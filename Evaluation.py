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
from pytorch_grad_cam import GradCAM
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
    elif model_type.lower() == "swin":
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

def run_evaluation(data_dir, result_folder, model_weights_path, num_classes, batch_size, seed, model_type):
    utils_setup.set_seed(seed)
    model, cam = build_model_and_cam(model_type, model_weights_path, num_classes)

    data_transforms = transforms.Compose([
        transforms.Resize((1200, 1200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(data_dir, 'test')
    categories = ['columnar', 'equiax']
    class_to_label = {"equiax": 0, "columnar": 1}
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
            image_probabilities_pos = []
            image_probabilities_neg = []

            for images, labels_batch, paths_batch in tqdm(data_loader_test, desc='Evaluating'):
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)

                targets = [ClassifierOutputTarget(int(pred.cpu().numpy())) for pred in predicted]
                grayscale_cams = cam(input_tensor=images, targets=targets)

                for i in range(len(labels_batch)):
                    grayscale_cam = grayscale_cams[i]
                    file_path = paths_batch[i]
                    pil_img = Image.open(file_path).convert('RGB')
                    pil_img = pil_img.resize((1200, 1200))
                    rgb_img = np.array(pil_img, dtype=np.float32) / 255.0
                    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                    base_name = os.path.basename(file_path)
                    fname_no_ext, ext = os.path.splitext(base_name)
                    save_path = os.path.join(experiment_result_folder,
                                             f"{fname_no_ext}_gradcam{ext}")
                    cv2.imwrite(save_path, cam_image_bgr)

                y_true.extend(labels_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                for i in range(len(labels_batch)):
                    pos_prob = probabilities[i, 1].item()
                    neg_prob = probabilities[i, 0].item()
                    image_results.append({
                        'Filename': paths_batch[i],
                        'True_Label': labels_batch[i].item(),
                        'Predicted_Label': predicted[i].item(),
                        'Pos_Prob': pos_prob,
                        'Neg_Prob': neg_prob
                    })
                    image_probabilities_pos.append(pos_prob)
                    image_probabilities_neg.append(neg_prob)

            overall_y_true.extend(y_true)
            overall_y_pred.extend(y_pred)

            subfolder_results_df = pd.DataFrame(image_results)
            subfolder_csv_path = os.path.join(experiment_result_folder, "image_results.csv")
            subfolder_results_df.to_csv(subfolder_csv_path, index=False)
            print(f"Classification results saved to {subfolder_csv_path}")

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

            x_indices = range(len(image_probabilities_pos))

            pos_ymin, pos_ymax = set_y_axis(image_probabilities_pos)
            plt.figure(figsize=(12, 6))
            plt.plot(x_indices, image_probabilities_pos, marker='o',
                     linestyle='-', color='blue')
            plt.ylim(pos_ymin, pos_ymax)
            plt.xlabel('Step (Image Index)', fontsize=14)
            plt.ylabel('Positive Probability', fontsize=14)
            plt.title('Positive Probability per Image', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_result_folder, "positive_probability_per_image.png"))
            plt.close()

            neg_ymin, neg_ymax = set_y_axis(image_probabilities_neg)
            plt.figure(figsize=(12, 6))
            plt.plot(x_indices, image_probabilities_neg, marker='o',
                     linestyle='-', color='red')
            plt.ylim(neg_ymin, neg_ymax)
            plt.xlabel('Step (Image Index)', fontsize=14)
            plt.ylabel('Negative Probability', fontsize=14)
            plt.title('Negative Probability per Image', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_result_folder, "negative_probability_per_image.png"))
            plt.close()

            print(f"Probability curves saved to {experiment_result_folder}")

    overall_accuracy = accuracy_score(overall_y_true, overall_y_pred)
    print(f"Overall model accuracy: {overall_accuracy * 100:.2f}%")

    overall_results_path = os.path.join(result_folder, "overall_results.csv")
    overall_results_df = pd.DataFrame({'Overall_Accuracy': [overall_accuracy]})
    overall_results_df.to_csv(overall_results_path, index=False)
    print(f"Overall accuracy saved to {overall_results_path}")

    overall_conf_matrix = confusion_matrix(overall_y_true, overall_y_pred, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(overall_conf_matrix, annot=True, fmt="d", cmap="Greens",
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
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
        get_evaluation = False,
        animation = False, 
        get_accuracy_per_video = False, 
        get_overall_balanced_accuracy = False
        ):

    # Call the evaluation process
    if get_evaluation is True:
        run_evaluation(DATA_DIR, RESULT_FOLDER, MODEL_WEIGHTS_PATH, num_classes, batch_size, seed, model_type)

    if animation is True:
        utils_analysis.generate_animations(RESULT_FOLDER, DATA_DIR)
    
    if get_accuracy_per_video is True:
        utils_analysis.calculate_accuracy_for_folders(RESULT_FOLDER)
    
    if get_overall_balanced_accuracy is True:
        utils_analysis.compute_overall_balanced_accuracy(RESULT_FOLDER)

if __name__ == "__main__":
    # Parameter settings
    DATA_DIR = r'/home/shun/Project/Grains-Classification/Dataset/Test_val'
    RESULT_FOLDER = r'/home/shun/Project/Grains-Classification/Result_test_2025.01/ResNet_frozen/evaluation'
    MODEL_WEIGHTS_PATH = r'/home/shun/Project/Grains-Classification/Result_test_2025.01/ST_frozen/checkpoints/best_model_epoch_18_val_loss_0.0006.pth'
    num_classes = 2
    batch_size = 1
    seed = 10086
    model_type = "swin"  # Options: "resnet" or "swin", remember to change the path of model weight.
    # Run evaluation
    get_evaluation = False

    # Animation
    animation = False

    # Accuracy per video
    get_accuracy_per_video = True

    # Overall balanced accuracy
    get_overall_balanced_accuracy = True

    main(DATA_DIR, RESULT_FOLDER, MODEL_WEIGHTS_PATH, num_classes, batch_size, seed, model_type, get_evaluation, animation, get_accuracy_per_video, get_overall_balanced_accuracy)
