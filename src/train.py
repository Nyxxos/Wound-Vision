import torch
import torch.nn as nn
from dataset import get_dataloaders, selected_classes
from model import wound_vision_model

# Running the model on either the GPU or the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Extracting the train, validation and test split from dataset.py
train_loader, val_loader, test_loader = get_dataloaders()
# Moving the model onto the device
model = wound_vision_model(len(selected_classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.fc.parameters(), lr = 0.001)

num_epochs = 50
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            max_value, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Val Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "wound_vision_best_val.pth")
