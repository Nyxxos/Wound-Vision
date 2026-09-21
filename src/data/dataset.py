import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from collections import Counter

data_path = r"C:\Users\Haroon\OneDrive\Desktop\Projects\Wound-Vision\wound-classification-dataset"
selected_classes = ["Abrasions", "Bruises", "Burns", "Cut", "Normal"]

train_transforms = v2.Compose([
     v2.Resize((224, 224)),
     v2.RandomHorizontalFlip(),
     v2.ToImage(),
     v2.ToDtype(torch.float32, scale = True),
     v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

val_transforms = v2.Compose([
     v2.Resize((224, 224)),
     v2.ToImage(),
     v2.ToDtype(torch.float32, scale = True),
     v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

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

def get_dataloaders(batch_size = 32):
    train_raw = ImageFolder(data_path, transform = train_transforms)
    val_raw = ImageFolder(data_path, transform = val_transforms) 

    selected_indices = [train_raw.class_to_idx[name] for name in selected_classes]
    new_label = {old_index: new_index for new_index, old_index in enumerate(selected_indices)}

    filtered_positions = [i for i, sample in enumerate(train_raw.samples) if sample[1] in selected_indices]
    filtered_labels = [train_raw.samples[i][1] for i in filtered_positions]
    print("Filtered labels:")
    print(Counter(filtered_labels))
    train_positions, val_test_positions = train_test_split(filtered_positions, test_size = 0.3, random_state = 42, stratify = filtered_labels)

    val_test_labels = [train_raw.samples[i][1] for i in val_test_positions]
    print("val_test labels:")
    print(Counter(val_test_labels))
    val_positions, test_positions = train_test_split(val_test_positions, test_size = 0.5, random_state = 42, stratify = val_test_labels)

    train_data = RemappedDataset(Subset(train_raw, train_positions), new_label)
    val_data = RemappedDataset(Subset(val_raw, val_positions), new_label)
    test_data = RemappedDataset(Subset(val_raw, test_positions), new_label)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

    return train_loader, val_loader, test_loader

print

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
