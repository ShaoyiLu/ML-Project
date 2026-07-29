import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt

def display_images(data, labels):
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
    for i, label in enumerate(unique_labels):
        idx = np.where(labels == label)[0][0]
        image = data[idx]
        axes[i].imshow(image.astype(np.uint8))
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()

def load_pth(file_path):
    raw_data = torch.load(file_path)
    return raw_data['data'].numpy(), raw_data['labels'].numpy()

if __name__ == '__main__':
    
    # Load and display images for each class in the dataset
    train_data, train_labels = load_pth('val_data_model_3.pth')
    print('Displaying images for all classes:')
    display_images(train_data, train_labels)