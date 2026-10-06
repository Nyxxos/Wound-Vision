import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_dataloaders, selected_classes
from model import wound_vision_model

model_path = r"C:\WoundVisionModels\wound_vision_v1.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, test_loader = get_dataloaders()
model = wound_vision_model(len(selected_classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location = device, weights_only = True))
criterion = nn.CrossEntropyLoss()
all_labels = []
all_predictions = []

# Test
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += (loss.item() * images.size(0))
        predicted = outputs.argmax(dim=1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predicted.cpu().tolist())

test_loss = test_loss / test_total
test_accuracy = test_correct / test_total

# Macro average was used here since my classes aren't evenly sized and dont want my smaller classes (cuts/burns) to be drowned out
precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
matrix = confusion_matrix(all_labels, all_predictions)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-score: {f1:.2%}")
print(classification_report(all_labels, all_predictions, target_names=selected_classes, digits=4, zero_division=0))
print(matrix)