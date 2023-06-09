import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 64
learning_rate = 1e-3
num_epochs = 10

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder('C:/Users/mnb35/Downloads/PYTHON/SIGN/CAR_CLASSIFIED', transform=transform)
test_dataset = datasets.ImageFolder('C:/Users/mnb35/Downloads/PYTHON/SIGN/TEST_CLASSIFIED', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

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


#Set parameters such as batch size, learning rate, and the number of epochs.
#Define image transformations, including converting images to grayscale, resizing them to 200*200 pixels, and converting them to tensors.
#Load the training and test datasets using the ImageFolder class and apply the defined transformations.
#Create data loaders for the training and test datasets to batch and load the data.
#Define the CRNN (Convolutional Recurrent Neural Network) model class, including convolutional layers, pooling layers, fully connected layers, LSTM layer, and output layer.
#Instantiate the CRNN model.
#Define the loss function (cross-entropy loss) and optimizer (Adam optimizer).
#Train the model by iterating over multiple epochs. For each iteration, retrieve images and labels from the data loader,perform forward pass, compute the loss, backpropagate, and optimize the parameters.
#Display the current training progress at specific steps (every 100 steps).
#After training, evaluate the model on the test dataset. Set the model to evaluation mode, disable gradient computation. Iterate over images and labels in the test data loader, make predictions, calculate classification accuracy.
