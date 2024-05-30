import numpy as np
import pandas as pd
from skimage import io
from skimage.color import rgb2gray
from ripser import ripser
import os
import concurrent.futures

# Paths
image_dir = 'data/raw_data/ISIC2018/test'
label_path = 'data/raw_data/ISIC2018/label_test.csv'
output_dir = 'data/raw_data/ISIC2018/test_ph'

# Load labels
labels = pd.read_csv(label_path)
label_dict = labels.set_index('image').to_dict('index')  # Convert row per image to dictionary

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to process each image
def process_image(filename):
    full_path = os.path.join(image_dir, filename)
    image = io.imread(full_path)

    # Convert to grayscale if it is a color image
    if len(image.shape) == 3:
        image = rgb2gray(image)

    # Flatten the image and sample points
    points = image.flatten()
    indices = np.random.choice(len(points), size=min(1000, len(points)), replace=False)
    sampled_points = points[indices].reshape(-1, 1)

    # Compute persistence diagrams
    result = ripser(sampled_points)
    diagrams = result['dgms']
    
    diagrams_with_group = []
    for i, dg in enumerate(diagrams):
        if dg.size > 0:  # Ensure the diagram is not empty
            # Create a one-hot encoded array for homology group
            # Assuming i = 0 for H0 and i = 1 for H1
            one_hot = np.zeros((dg.shape[0], 2))  # Two columns for two groups
            one_hot[:, i] = 1  # Set the i-th column to 1
            
            # Concatenate the original diagram points with the one-hot encoding
            dg_with_group = np.hstack((dg, one_hot))
            diagrams_with_group.append(dg_with_group)
        else:
            # Handle empty diagrams
            diagrams_with_group.append(np.empty((0, 4)))

    # Placeholder for generating a landscape
    landscapes = np.random.rand(10, 100)  # Random data simulating landscapes

    # Extract the label for the image
    image_key = os.path.splitext(filename)[0]
    if image_key in label_dict:
        label = label_dict[image_key]
    else:
        label = None  # In case there is no label entry for some image

    # Bundle the diagram, landscape, label, and image path
    result = {
        'diagram': diagrams_with_group,
        'landscape': landscapes,
        'label': label,
        'image_path': full_path  # Include the full path to the image
    }

    # Save to a single NPY file
    output_file_path = os.path.join(output_dir, f"{image_key}.npy")
    np.save(output_file_path, result)

    return f"Processed and saved {filename}"

# Main function to execute parallel processing
def main():
    files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, f) for f in files]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()

print("All images have been processed with diagrams, landscapes, and labels saved.")
