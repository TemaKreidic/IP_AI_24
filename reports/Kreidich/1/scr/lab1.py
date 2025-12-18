from torchvision import transforms
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random


data_dir = './data'

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=train_transform),
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=test_transform),
    batch_size=64, shuffle=False
)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

def train(model, loader):
    model.train()
    loss_sum, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return loss_sum / len(loader), 100 * correct / total

def test(model, loader):
    model.eval()
    loss_sum, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return loss_sum / len(loader), 100 * correct / total


num_epochs = 10
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(num_epochs):
    tr_loss, tr_acc = train(model, train_loader)
    te_loss, te_acc = test(model, test_loader)

    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    train_accs.append(tr_acc)
    test_accs.append(te_acc)

    print(f'Epoch {epoch+1}/{num_epochs} | '
          f'Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.2f}% | '
          f'Test Loss: {te_loss:.4f}, Test Acc: {te_acc:.2f}%')


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(test_accs, label='Test')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()


def show_random_image(model, loader):
    model.eval()
    images, labels = next(iter(loader))
    idx = random.randint(0, images.size(0) - 1)

    image = images[idx].unsqueeze(0).to(device)
    label = labels[idx].item()

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(1).item()

    img = image.cpu().squeeze()
    img = img * 0.5 + 0.5

    plt.imshow(img.numpy(), cmap='gray')
    plt.title(f'Predicted: {pred}, Actual: {label}')
    plt.axis('off')
    plt.show()

show_random_image(model, test_loader)
