import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def wound_vision_model(num_classes):
    model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
    # Freeze all layers as the dataset is small (~ 600 images) so only the new final layer below should learn to avoid overfitting
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model




