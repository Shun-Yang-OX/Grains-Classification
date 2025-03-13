import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import seaborn as sns


def create_animation_for_subfolder(subfolder_path, save_animation_path, orig_images_base_dir, with_heatmap=True):
    """
    For a given subfolder, read the prediction results and class probabilities from image_results.csv,
    and combine the corresponding original images and (optional) Grad-CAM heatmaps to generate an animation (.mp4).
    If with_heatmap is True, the animation will include the original image, Grad-CAM heatmap, and probability curves;
    otherwise, it will only include the original image and probability curves.
    """
    print(f"Processing folder: {subfolder_path}")
    
    # Check if the CSV file exists
    csv_path = os.path.join(subfolder_path, "image_results.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}, skipping this folder.")
        return

    # Read the CSV file
    results_df = pd.read_csv(csv_path)
    
    # Check that the required columns exist (including the new Prob_Class_3 column)
    required_columns = [
        'Filename', 'True_Label', 'Predicted_Label',
        'Prob_Class_0', 'Prob_Class_1', 'Prob_Class_2', 'Prob_Class_3'
    ]
    if not all(col in results_df.columns for col in required_columns):
        print(f"Missing required columns in {csv_path}, cannot generate animation. Skipping this folder.")
        return

    # Obtain class probabilities
    prob_class_0 = results_df['Prob_Class_0'].tolist()
    prob_class_1 = results_df['Prob_Class_1'].tolist()
    prob_class_2 = results_df['Prob_Class_2'].tolist()
    prob_class_3 = results_df['Prob_Class_3'].tolist()

    # Form a list of absolute paths for the original images
    relative_image_paths = results_df['Filename'].tolist()
    image_paths = [os.path.join(orig_images_base_dir, rel_path) for rel_path in relative_image_paths]

    # If Grad-CAM heatmaps are needed, construct the corresponding path list (assuming file suffix is _gradcam)
    if with_heatmap:
        gradcam_paths = [
            os.path.join(
                subfolder_path,
                f"{os.path.splitext(os.path.basename(p))[0]}_gradcam{os.path.splitext(p)[1]}"
            )
            for p in image_paths
        ]

    num_frames = len(image_paths)
    if num_frames == 0:
        print(f"No valid image data found in {subfolder_path}, skipping.")
        return

    # Create different numbers of subplots depending on whether heatmap is needed
    if with_heatmap:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax_orig, ax_heatmap, ax_prob = axes
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax_orig, ax_prob = axes

    # Subplot 1: Display the original image
    orig_display = ax_orig.imshow(np.zeros((600, 600, 3), dtype=np.uint8))
    ax_orig.axis('off')
    ax_orig.set_title('Original Image')

    # Subplot 2 (only if with_heatmap=True): Display the Grad-CAM heatmap
    if with_heatmap:
        heatmap_display = ax_heatmap.imshow(np.zeros((600, 600, 3), dtype=np.uint8))
        ax_heatmap.axis('off')
        ax_heatmap.set_title('Grad-CAM Heatmap')

    # Last subplot: Display probability curves (adding an extra curve)
    line0, = ax_prob.plot([], [], 'b-o', label='Equiaxed')
    line1, = ax_prob.plot([], [], 'r-o', label='Columnar')
    line2, = ax_prob.plot([], [], 'g-o', label='Background')
    line3, = ax_prob.plot([], [], 'm-o', label='IMC')
    ax_prob.set_xlim(0, num_frames - 1)
    ax_prob.set_ylim(0, 1)
    ax_prob.set_xlabel('Frame')
    ax_prob.set_ylabel('Probability')
    ax_prob.set_title('Microstructure in Images')
    ax_prob.legend()

    # Data containers for updated values
    x_data = []
    prob0_data = []
    prob1_data = []
    prob2_data = []
    prob3_data = []

    def update(frame_idx):
        # Read and update the original image
        orig_img_path = image_paths[frame_idx]
        try:
            orig_img = Image.open(orig_img_path).convert('RGB').resize((600, 600))
        except Exception as e:
            print(f"Error reading original image: {e}")
            return (orig_display, heatmap_display, line0, line1, line2, line3) if with_heatmap else (orig_display, line0, line1, line2, line3)

        orig_display.set_data(np.array(orig_img))

        # If heatmap is needed, read and update the heatmap image
        if with_heatmap:
            gradcam_img_path = gradcam_paths[frame_idx]
            try:
                heatmap_img = Image.open(gradcam_img_path).resize((600, 600))
            except Exception as e:
                print(f"Error reading Grad-CAM image: {e}")
                heatmap_img = np.zeros((600, 600, 3), dtype=np.uint8)
            heatmap_display.set_data(np.array(heatmap_img))

        # Update probability curve data
        x_data.append(frame_idx)
        prob0_data.append(prob_class_0[frame_idx])
        prob1_data.append(prob_class_1[frame_idx])
        prob2_data.append(prob_class_2[frame_idx])
        prob3_data.append(prob_class_3[frame_idx])

        line0.set_data(x_data, prob0_data)
        line1.set_data(x_data, prob1_data)
        line2.set_data(x_data, prob2_data)
        line3.set_data(x_data, prob3_data)

        if with_heatmap:
            return orig_display, heatmap_display, line0, line1, line2, line3
        else:
            return orig_display, line0, line1, line2, line3

    # Generate animation using FuncAnimation
    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)

    # Save the animation as an MP4 file
    try:
        writer = FFMpegWriter(fps=30)
        anim.save(save_animation_path, writer=writer)
        print(f"Animation saved to: {save_animation_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")

    plt.close()
    print(f"Finished processing folder: {subfolder_path}\n")


def generate_animations(root_result_dir, orig_images_base_dir, with_heatmap=True):
    """
    Traverse all subfolders under root_result_dir that contain image_results.csv,
    generate an animation for each subfolder, and save it as animation.mp4.
    If with_heatmap is True, the animation will include the Grad-CAM heatmap; otherwise, it will not.
    """
    valid_subfolders = [
        os.path.join(root_result_dir, folder) 
        for folder in os.listdir(root_result_dir)
        if os.path.isdir(os.path.join(root_result_dir, folder)) and 
           os.path.exists(os.path.join(root_result_dir, folder, "image_results.csv"))
    ]

    for subfolder_path in tqdm(valid_subfolders, desc="Processing folders"):
        save_path = os.path.join(subfolder_path, "animation.mp4")
        create_animation_for_subfolder(subfolder_path, save_path, orig_images_base_dir, with_heatmap=with_heatmap)


def calculate_accuracy_for_folders(parent_folder):
    """
    Calculate the accuracy within each subfolder and aggregate the results into an accuracy_summary.csv file.
    This is applicable when each subfolder contains image_results.csv and other CSV files.
    """
    results = []
    
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Find all CSV files in the current subfolder
            csv_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.csv')]

            for csv_file in csv_files:
                csv_path = os.path.join(subfolder_path, csv_file)
                try:
                    data = pd.read_csv(csv_path)
                    if 'True_Label' in data.columns and 'Predicted_Label' in data.columns:
                        accuracy = (data['True_Label'] == data['Predicted_Label']).mean()
                        results.append({
                            'Subfolder': subfolder,
                            'CSV_File': csv_file,
                            'Accuracy': accuracy
                        })
                except Exception as e:
                    print(f"Cannot process file {csv_path}: {e}")
    
    if len(results) == 0:
        print("No usable CSV found in subfolders for accuracy statistics.")
        return

    result_df = pd.DataFrame(results)
    output_path = os.path.join(parent_folder, "accuracy_summary.csv")
    result_df.to_csv(output_path, index=False)
    
    print(f"Subfolder accuracy statistics completed, results saved to: {output_path}")


def compute_overall_balanced_accuracy(result_base_dir):
    """
    Traverse all image_results.csv files under the specified root directory (including subdirectories),
    extract the true and predicted labels for the four classes (0=Equiaxed, 1=Columnar, 2=Background, 3=IMC),
    and compute and output the following metrics:
       - Four-class macro balanced accuracy (balanced_accuracy_score).
       - A 4x4 confusion matrix, plotted using seaborn and saved.
       - The accuracy for each class (diagonal element divided by the row sum).
    """
    # Get all files named image_results.csv (including subdirectories)
    csv_pattern = os.path.join(result_base_dir, '**', 'image_results.csv')
    csv_files = glob.glob(csv_pattern, recursive=True)

    all_true_labels = []
    all_pred_labels = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if 'True_Label' in df.columns and 'Predicted_Label' in df.columns:
            all_true_labels.extend(df['True_Label'].tolist())
            all_pred_labels.extend(df['Predicted_Label'].tolist())
        else:
            print(f"{csv_file} is missing True_Label or Predicted_Label columns, skipping.")

    if not all_true_labels or not all_pred_labels:
        print("No valid label data collected, unable to compute overall metrics.")
        return

    # 1) Four-class macro balanced accuracy
    macro_accuracy = balanced_accuracy_score(all_true_labels, all_pred_labels)
    print(f"Four-class macro-balanced accuracy: {macro_accuracy * 100:.2f}%")

    # 2) Compute the 4x4 confusion matrix
    class_labels = [0, 1, 2, 3]  # 0=Equiaxed, 1=Columnar, 2=Background, 3=IMC
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=class_labels)

    print("Four-class confusion matrix:")
    print(conf_matrix)

    # 3) Calculate accuracy for each class: diagonal element / row sum
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    per_class_accuracy = conf_matrix.diagonal() / row_sums.reshape(-1)

    label_names = ["Equiaxed (0)", "Columnar (1)", "Background (2)", "IMC (3)"]
    for i, acc in enumerate(per_class_accuracy):
        print(f"Accuracy for {label_names[i]}: {acc * 100:.2f}%")

    # 4) Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names,
                yticklabels=label_names)
    plt.title("Overall Four-class Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    conf_matrix_path = os.path.join(result_base_dir, "overall_confusion_matrix_4_classes.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"Four-class confusion matrix saved to: {conf_matrix_path}")
