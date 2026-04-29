import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import VideoDataset
from src.model import AccidentTransformer

# datasets
train_data = VideoDataset("dataset_clips/train")
val_data = VideoDataset("dataset_clips/val")

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=2)

# model
model = AccidentTransformer()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# training loop
for epoch in range(5):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            output = model(x)
            preds = torch.argmax(output, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")

# save model
torch.save(model.state_dict(), "model.pth")
print("Model saved!")