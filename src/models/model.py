import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def wound_vision_model(num_classes):
    model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
    for parameters in model.parameters():
        parameters.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model




