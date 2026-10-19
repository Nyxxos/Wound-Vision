import torch
from PIL import Image
from dataset import selected_classes, val_transforms
from model import wound_vision_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"C:\WoundVisionModels\wound_vision_v1.pth"
model = wound_vision_model(len(selected_classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location = device, weights_only = True))

model.eval()
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = val_transforms(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim = 1)
        top_probs, top_indices = torch.topk(probabilities, k = 3, dim = 1)

    top_results = []
    for score, index in zip(top_probs[0], top_indices[0]):
        top_results.append({"class": selected_classes[index.item()], "confidence": score.item()})
        
    predicted_class = top_results[0]["class"]
    confidence = top_results[0]["confidence"]

    return predicted_class, confidence, top_results