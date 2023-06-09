"""
This script trains a Convolutional Recurrent Neural Network (CRNN) model for a specific task.

Functionality:
- Sets parameters such as batch size, learning rate, and number of epochs.
- Loads training and test datasets and applies preprocessing transformations.
- Defines the CRNN model architecture.
- Trains the model by optimizing the parameters using backpropagation.
- Evaluates the trained model on the test dataset and calculates classification accuracy.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 32
learning_rate = 1e-3
num_epochs = 30

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])


train_dataset = datasets.ImageFolder('C:/Users/mnb35/Downloads/PYTHON/SIGN/CAR_CLASSIFIED', transform=transform)
test_dataset = datasets.ImageFolder('C:/Users/mnb35/Downloads/PYTHON/SIGN/TEST_CLASSIFIED', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#from torch.utils.data import random_split

#total_samples = len(train_dataset)
#split_ratio = 0.2
#test_samples = int(total_samples * split_ratio)
#train_samples = total_samples - test_samples

#train_dataset, test_dataset = random_split(train_dataset, [train_samples, test_samples])

#train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*50*50, 128)

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = self.fc2(x.squeeze(1))
        return x

model = CRNN()
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))
