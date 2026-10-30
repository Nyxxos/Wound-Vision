import torch
import numpy as np
from PIL import Image
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from inference import model, device
from dataset import val_transforms, selected_classes

def gradcam_heatmap(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = val_transforms(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_(True)
    rgb_image = image.resize((224, 224))
    rgb_image = np.array(rgb_image, dtype = np.float32) / 255.0
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = probabilities.max(dim = 1)
    predicted_class = selected_classes[predicted.item()]
    true_class = Path(image_path).parent.name
    correct = predicted_class == true_class
    target_layers = [model.layer4[-1]]

    with GradCAM(model = model, target_layers = target_layers) as cam:
        grayscale_cam = cam(input_tensor = image_tensor, targets=None)

    grayscale_cam = grayscale_cam[0]
    heatmap = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return (Image.fromarray(heatmap), predicted_class, confidence.item(), true_class, correct)

heatmap, prediction, confidence, true_class, correct = gradcam_heatmap(r"C:\Users\Haroon\OneDrive\Desktop\Projects\Wound-Vision\wound-classification-dataset\Burns\burns (1).jpg")
print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2%}")
print(f"True class: {true_class}")
print(f"Correct: {correct}")
heatmap.show()