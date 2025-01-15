import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

def create_animation_for_subfolder(subfolder_path, save_animation_path, orig_images_base_dir):
    print(f"Starting processing folder: {subfolder_path}")
    
    # Check if the CSV file exists
    csv_path = os.path.join(subfolder_path, "image_results.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}, skipping {subfolder_path}")
        return

    # Read CSV file
    results_df = pd.read_csv(csv_path)

    # Assume that 'Filename' column contains paths relative to the original image base directory
    relative_image_paths = results_df['Filename'].tolist()
    # Construct full path to original images
    image_paths = [os.path.join(orig_images_base_dir, rel_path) for rel_path in relative_image_paths]

    # Construct corresponding Grad-CAM image paths, assuming they are located in the current subfolder
    gradcam_paths = [
        os.path.join(subfolder_path, f"{os.path.splitext(os.path.basename(p))[0]}_gradcam{os.path.splitext(p)[1]}")
        for p in image_paths
    ]

    # Choose probability curve based on class
    true_label = results_df['True_Label'].iloc[0] if 'True_Label' in results_df.columns else None
    if true_label == 1:
        prob_type = 'Positive Probability'
        probs = results_df['Pos_Prob'].tolist()
        line_color = 'b-o'
    elif true_label == 0:
        prob_type = 'Negative Probability'
        probs = results_df['Neg_Prob'].tolist()
        line_color = 'r-o'
    else:
        prob_type = 'Probability'
        probs = results_df['Pos_Prob'].tolist()  # Default use positive class probability
        line_color = 'b-o'

    num_frames = len(image_paths)
    if num_frames == 0:
        print(f"No image data in {subfolder_path}, skipping.")
        return

    # Get folder name for title display
    folder_name = os.path.basename(subfolder_path)

    # Set animation layout: three columns for original image, heatmap and probability curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Experiment: {folder_name}", fontsize=16)
    ax_orig, ax_heatmap, ax_prob = axes

    # Initialize left subplot for displaying original image
    orig_display = ax_orig.imshow(np.zeros((600, 600, 3), dtype=np.uint8))
    ax_orig.axis('off')
    ax_orig.set_title('Original Image')

    # Initialize middle subplot for displaying Grad-CAM heatmap
    heatmap_display = ax_heatmap.imshow(np.zeros((600, 600, 3), dtype=np.uint8))
    ax_heatmap.axis('off')
    ax_heatmap.set_title('Grad-CAM Heatmap')

    # Initialize right subplot for displaying probability curve
    line, = ax_prob.plot([], [], line_color, label=prob_type)
    ax_prob.set_xlim(0, num_frames - 1)
    ax_prob.set_ylim(0, 1)
    ax_prob.set_xlabel('Frame Index')
    ax_prob.set_ylabel('Probability')
    ax_prob.set_title(f'{prob_type} Over Frames')
    ax_prob.legend()

    # Store displayed probability points to plot cumulative curve
    displayed_probs = []
    x_data = []

    def update(frame):
        # Load current frame's original image and corresponding heatmap
        try:
            orig_img = Image.open(image_paths[frame]).convert('RGB').resize((600, 600))
            heatmap_img = Image.open(gradcam_paths[frame]).resize((600, 600))
        except Exception as e:
            print(f"Error loading image: {e}")
            return orig_display, heatmap_display, line

        # Update original image and heatmap display
        orig_display.set_data(np.array(orig_img))
        heatmap_display.set_data(np.array(heatmap_img))

        # Update probability curve data
        x_data.append(frame)
        displayed_probs.append(probs[frame])
        line.set_data(x_data, displayed_probs)

        return orig_display, heatmap_display, line

    anim = FuncAnimation(fig, update, frames=num_frames, interval=500, blit=True)

    # Use FFMpegWriter to save animation as MP4 video
    try:
        writer = FFMpegWriter(fps=2)
        anim.save(save_animation_path, writer=writer)
        print(f"Video saved to {save_animation_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
    plt.close()
    print(f"Finished processing folder: {subfolder_path}\n")

def generate_animations(root_result_dir, orig_images_base_dir):
    """
    Traverse each subfolder under root_result_dir to generate animation videos.
    Parameters:
        root_result_dir: Root directory containing subfolders, CSV files and Grad-CAM images.
        orig_images_base_dir: Root directory of the original images.
    """
    # Collect all valid subfolders containing image_results.csv
    valid_subfolders = [
        os.path.join(root_result_dir, folder) 
        for folder in os.listdir(root_result_dir)
        if os.path.isdir(os.path.join(root_result_dir, folder)) and 
           os.path.exists(os.path.join(root_result_dir, folder, "image_results.csv"))
    ]

    # Display progress bar using tqdm
    for subfolder_path in tqdm(valid_subfolders, desc="Processing folders"):
        save_path = os.path.join(subfolder_path, "animation.mp4")
        create_animation_for_subfolder(subfolder_path, save_path, orig_images_base_dir)

def calculate_accuracy_for_folders(parent_folder):
    # List to store results
    results = []
    
    # Traverse all subfolders in parent folder
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        
        # Ensure current item is a folder
        if os.path.isdir(subfolder_path):
            # Search for CSV files in subfolder
            csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
            
            # If CSV files exist, process each found file
            for csv_file in csv_files:
                csv_path = os.path.join(subfolder_path, csv_file)
                try:
                    # Read CSV file
                    data = pd.read_csv(csv_path)
                    
                    # Ensure that the file has the required columns
                    if 'True_Label' in data.columns and 'Predicted_Label' in data.columns:
                        # Calculate accuracy
                        accuracy = (data['True_Label'] == data['Predicted_Label']).mean()
                        
                        # Save result
                        results.append({
                            'Subfolder': subfolder,
                            'Accuracy': accuracy
                        })
                except Exception as e:
                    print(f"Unable to process file {csv_path}: {e}")
    
    # Convert to DataFrame and save to CSV file
    result_df = pd.DataFrame(results)
    output_path = os.path.join(parent_folder, "accuracy_summary.csv")
    result_df.to_csv(output_path, index=False)
    
    print(f"Statistics complete, results saved to: {output_path}")

def compute_overall_balanced_accuracy(result_base_dir):
    """
    Traverse all subdirectories under the specified root directory, read CSV files containing labels,
    calculate and output the overall macro-average accuracy, negative class accuracy, and positive class accuracy.
    
    Parameters:
        result_base_dir (str): Root directory path containing evaluation results.
    """
    # Construct path pattern to search for CSV files
    csv_pattern = os.path.join(result_base_dir, '**', 'image_results.csv')
    csv_files = glob.glob(csv_pattern, recursive=True)

    all_true_labels = []
    all_pred_labels = []

    # Traverse each found CSV file, read and extract labels
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if 'True_Label' in df.columns and 'Predicted_Label' in df.columns:
            all_true_labels.extend(df['True_Label'].tolist())
            all_pred_labels.extend(df['Predicted_Label'].tolist())
        else:
            print(f"File {csv_file} does not contain the required label columns.")

    # If valid label data is found, calculate metrics
    if all_true_labels and all_pred_labels:
        # Calculate macro-average accuracy
        macro_accuracy = balanced_accuracy_score(all_true_labels, all_pred_labels)
        print(f"Overall Macro-Average Accuracy: {macro_accuracy * 100:.2f}%")

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=[0, 1])
        # Confusion matrix format:
        # [[TN, FP],
        #  [FN, TP]]

        # If confusion matrix is 2x2, calculate negative and positive class accuracy
        if conf_matrix.shape[0] == 2:
            neg_total = conf_matrix[0].sum()
            pos_total = conf_matrix[1].sum()

            if neg_total > 0:
                neg_accuracy = conf_matrix[0, 0] / neg_total
                print(f"Negative Class Accuracy: {neg_accuracy * 100:.2f}%")
            else:
                print("No negative class samples in the global dataset.")

            if pos_total > 0:
                pos_accuracy = conf_matrix[1, 1] / pos_total
                print(f"Positive Class Accuracy: {pos_accuracy * 100:.2f}%")
            else:
                print("No positive class samples in the global dataset.")
        else:
            print("Warning: Confusion matrix does not contain the expected two classes.")
    else:
        print("No valid label data found, unable to calculate macro-average accuracy.")
