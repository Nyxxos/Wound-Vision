import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from PIL import Image
import matplotlib.pyplot as plt
import random

data_path = r"C:\Users\Haroon\OneDrive\Desktop\Projects\Wound-Vision\wound-classification-dataset"

data_transforms = v2.Compose([
     v2.Resize((224, 224)),
     v2.ToImage(),
     v2.ToDtype(torch.float32, scale = True),
     v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

selected_classes = ["Abrasions", "Bruises", "Burns", "Cut", "Normal"]
data = ImageFolder(data_path, transform = data_transforms)
classes = data.classes

selected_indices = [data.class_to_idx[class_name] for class_name in selected_classes]
filtered_positions = [i for i, sample in enumerate(data.samples) if sample[1] in selected_indices]
filtered_data = Subset(data, filtered_positions)
print(f"Total images kept: {len(filtered_data)}")

for class_name in selected_classes:
    class_samples = [sample for sample in data.samples if sample[1] == data.class_to_idx[class_name]]
    random_path, _ = random.choice(class_samples)
    image = Image.open(random_path)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis("off")
    plt.show()

new_label = {old_index: new_index for new_index, old_index in enumerate(selected_indices)}

class RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, label_map):
        self.subset = subset
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, old_label = self.subset[index]
        label_remap = self.label_map[old_label]
        return image, label_remap

remapped_data = RemappedDataset(filtered_data, new_label)

data_loader = DataLoader(remapped_data, batch_size = 32, shuffle = True)
for batch_images, batch_labels in data_loader:
    print(batch_images.shape)
    print(batch_labels.shape)
    print(batch_labels)
    break