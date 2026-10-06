import torch
import torch.nn as nn
from dataset import get_dataloaders, selected_classes
from model import wound_vision_model

model_path = r"C:\WoundVisionModels\wound_vision_v1.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
train_loader, val_loader, _ = get_dataloaders()
model = wound_vision_model(len(selected_classes)).to(device)
criterion = nn.CrossEntropyLoss()

# Only the new classifier is being trained because the ResNet backbone is frozen
optimiser = torch.optim.Adam(model.fc.parameters(), lr=0.001)
num_epochs = 20
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        train_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss / train_total
    train_accuracy = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2%} | "
        f"Validation Loss: {val_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.2%}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), model_path)
        print(f"Model saved (validation accuracy: {best_val_accuracy:.2%})")

print("\nTraining complete.")
print(f"Best model saved to: {model_path}")