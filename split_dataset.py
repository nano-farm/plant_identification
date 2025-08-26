import os
import random
import shutil

# --- CONFIGURATION ---
# Set the path to the directory containing your chili class folders
source_dir = "data/train/chili" 

# Set the path to the directory where the validation files will be moved
dest_dir = "data/valid/chili"   

# Set the percentage of images to move to the validation set (e.g., 0.2 for 20%)
split_ratio = 0.2
# --- END CONFIGURATION ---

print(f"Starting dataset split. Moving {split_ratio*100}% of images from '{source_dir}' to '{dest_dir}'.")

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Iterate through each class subfolder in the source directory
for class_folder in os.listdir(source_dir):
    source_class_path = os.path.join(source_dir, class_folder)
    
    if os.path.isdir(source_class_path):
        dest_class_path = os.path.join(dest_dir, class_folder)
        os.makedirs(dest_class_path, exist_ok=True)
        
        # Get a list of all image files in the class folder
        files = [f for f in os.listdir(source_class_path) if os.path.isfile(os.path.join(source_class_path, f))]
        
        # Shuffle the list of files randomly
        random.shuffle(files)
        
        # Determine the number of files to move
        num_to_move = int(len(files) * split_ratio)
        
        # Select the files to move
        files_to_move = files[:num_to_move]
        
        # Move each selected file
        for file_name in files_to_move:
            shutil.move(os.path.join(source_class_path, file_name), os.path.join(dest_class_path, file_name))
            
        print(f"Moved {len(files_to_move)} files from '{class_folder}' to validation set.")

print("\nDataset split complete!")