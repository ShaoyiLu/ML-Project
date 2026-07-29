import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
class CAS771Dataset(Dataset):
    def __init__(self, data, labels, transform=False):
        self.data = data
        self.labels = labels
        self.transform = ToTensor() if transform else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform and not isinstance(img, torch.Tensor):
            img = self.transform(img)
        return img, label

def load_data(data_path):
    raw_data = torch.load(data_path)
    data = raw_data['data']
    #print(data.shape)
    labels = raw_data['labels']

    indices = raw_data['indices'] #indice is the idx of the data in the original dataset (CIFAR100)
    return data, labels, indices

def load_class_names(filepath):
    with open(filepath, 'r') as file:
        classes = [line.strip() for line in file]
    return classes

def remap_labels(labels, class_mapping):
    return [class_mapping[label] for label in labels]

if __name__ == '__main__':
    num_classes = 5
    train_data, labels, idx = load_data('model1_test_supercls.pth')
    classes = load_class_names('Task1_data/cifar100_classes.txt')
    my_dataset = CAS771Dataset(train_data, labels, transform=False)
    dataloader = DataLoader(my_dataset, batch_size=5, shuffle=True)

    # Get the first batch of images and labels
    images, labels = next(iter(dataloader))
    #print(images.shape)


    # Display the images and labels
    my_class = [classes[label] for label in labels] #match the class name with the label
    grid_img = make_grid(images, nrow=8)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(", ".join(my_class))
    plt.show()
