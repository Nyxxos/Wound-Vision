import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from PIL import Image
import matplotlib.pyplot as plt

data_path = r"C:\Users\Haroon\OneDrive\Desktop\Projects\Wound-Vision\wound-classification-dataset"

data_transforms = v2.Compose([
     v2.Resize((224, 224)),
     v2.ToImage(),
     v2.ToDtype(torch.float32, scale = True),
     v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

data = ImageFolder(data_path, transform = data_transforms)

classes = data.classes

for i, wound_class in enumerate(classes):
     print(f"{i} - {wound_class}")
     class_images = [sample for sample in data.samples if sample[1] ==i]
     image_path = class_images[0][0]
     image = Image.open(image_path)
     plt.imshow(image)
     plt.title(wound_class)
     plt.axis("off")
     plt.show()

data_loader = DataLoader(data, batch_size = 32, shuffle = True)
for batch_images, batch_labels in data_loader:
    print(batch_images.shape)
    print(batch_labels.shape)
    break