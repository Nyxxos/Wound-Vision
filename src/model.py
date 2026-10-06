import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def wound_vision_model(num_classes):
    model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
    # Freeze the model so it keeps its pretrained knowledge and doesnt overfit the small dataset (~ 600 images)
    for param in model.parameters():
        param.requires_grad = False
    # Replace ImageNets 1000-class classifier with out wound classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model



