import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm  

# Import custom modules
import Model
import utils

# Set the device, prioritizing GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom loader to convert grayscale to RGB
def grayscale_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')  # Convert image to RGB (three-channel)

def main(data_dir, result_folder, model_weights_path, num_classes, batch_size, seed):
    utils.set_seed(seed)
    model = Model.build_resnet152_for_xray(num_classes = 2).to(device)
    checkpoint = torch.load(model_weights_path, map_location=device)

    # Adjust keys, remove 'module.' prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Data Loading
    total_correct = 0
    total_samples = 0

    # Lists to store results for CSV and confusion matrix
    y_true = []
    y_pred = []
    filenames = []
    tp_files, tn_files, fp_files, fn_files = [], [], [], []  # Track all classifications

    # Define data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((1200, 1200)),  # Adjust the size if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])  # Adjust mean and std for 1 channel
    ])

    # Create test dataset and DataLoader with the custom loader
    test_dir = os.path.join(data_dir, 'test')  # Adjust the subdirectory name if different

    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms, loader=grayscale_loader,target_transform=lambda x: 1 - x)
    print("Test Dataset Classes:", test_dataset.classes)

    

    data_loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Get all file paths from the dataset
    all_paths = [path for path, _ in test_dataset.imgs]

    # Initialize the progress bar
    total_batches = len(data_loader_test)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader_test, desc='Evaluating', total=total_batches)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Track correct and total samples
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Append results for analysis
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Calculate the global indices of the samples in this batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + images.size(0)
            batch_filenames = all_paths[start_idx:end_idx]
            filenames.extend(batch_filenames)

            # Track all classification cases
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                file_path = batch_filenames[i]

                # Classify cases as TP, TN, FP, FN
                if pred_label == 1 and true_label == 1:
                    tp_files.append(file_path)
                elif pred_label == 0 and true_label == 0:
                    tn_files.append(file_path)
                elif pred_label == 1 and true_label == 0:
                    fp_files.append(file_path)
                elif pred_label == 0 and true_label == 1:
                    fn_files.append(file_path)

    # Calculate and print test accuracy
    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Create confusion matrix and save it as an image
    all_classes = [0, 1]
    conf_matrix = confusion_matrix(y_true, y_pred,labels=all_classes)
    print("Confusion Matrix:")
    print(conf_matrix)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues",cbar=True,
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes,
                annot_kws={"size": 20},
                linewidths=0.5, linecolor='black',
                vmin=0, vmax=np.max(conf_matrix))
    plt.xlabel("Predicted Label", fontsize=18)
    plt.ylabel("True Label", fontsize=18)
    plt.title("Confusion Matrix", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # 手动添加注释
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j + 0.5, i + 0.5, str(conf_matrix[i, j]), 
                    horizontalalignment='center', verticalalignment='center', color="black", fontsize=16)

    plt.tight_layout()
    
    conf_matrix_path = os.path.join(result_folder, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)

    # Calculate percentages for TP, TN, FP, and FN
    tp_percent = len(tp_files) / total_samples * 100
    tn_percent = len(tn_files) / total_samples * 100
    fp_percent = len(fp_files) / total_samples * 100
    fn_percent = len(fn_files) / total_samples * 100

    # Print the percentages
    print(f"TP Percentage: {tp_percent:.2f}%")
    print(f"TN Percentage: {tn_percent:.2f}%")
    print(f"FP Percentage: {fp_percent:.2f}%")
    print(f"FN Percentage: {fn_percent:.2f}%")

    # Save all results to CSV
    df = pd.DataFrame({
        'Filename': tp_files + tn_files + fp_files + fn_files,
        'Type': ['TP'] * len(tp_files) + ['TN'] * len(tn_files) +
                ['FP'] * len(fp_files) + ['FN'] * len(fn_files)
    })
    csv_path = os.path.join(result_folder, "test_results_full.csv")
    df.to_csv(csv_path, index=False)
    print(f"Full test results CSV saved to {csv_path}")

# Entry point
if __name__ == "__main__":
    DATA_DIR =  r'/home/shun/Project/Grains-Classification/Dataset/classifier_accuracy_test_final'
    RESULT_FOLDER = r'/home/shun/Project/Grains-Classification/Result_test/ResNet_unfrozen/global_result'
    MODEL_WEIGHTS_PATH = r'/home/shun/Project/Grains-Classification/Result_test/ResNet_unfrozen/ResNet_unfrozen_checkpoints/best_model_epoch_12_val_loss_0.0000.pth'  # Update with your model path
    num_classes = 2
    batch_size = 32
    seed = 10086

    main(DATA_DIR, RESULT_FOLDER, MODEL_WEIGHTS_PATH, num_classes, batch_size, seed)
