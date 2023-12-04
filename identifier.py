import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .custom_torch_functions import CustomTorch

num_classes = 10


class Plant_Identifier(nn.Module):
    def __init__(self):
        pass

    def forward_pass(self, d):
        d = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)(d)
        d = nn.BatchNorm2d(16)(d)
        d = nn.ReLU()(d)
        d = nn.MaxPool2d(kernel_size=2)(d)
        
        d = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        d = nn.BatchNorm2d(32)(d)
        d = nn.ReLU()(d)
        d = nn.MaxPool2d(kernel_size=2)(d)

        d = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        d = nn.BatchNorm2d(64)(d)
        d = nn.ReLU()(d)
        d = nn.MaxPool2d(kernel_size=2)(d)

        d = d.view(d.size(0)-1)
        d = nn.Linear(64 * 28 *28, 256)(d)
        d = nn.ReLU()(d)
        d = nn.Linear(256, num_classes)(d)
        return d

        






class Plant_Identifier_Custom(nn.Module):
    def __init__(self):
        pass

    def forward_pass(self, d):
        d = CustomTorch.custom_conv2d(d, 3, 16, kernel_size=3, stride=1, padding=1)
        d = CustomTorch.custom_batch_norm2d(d, 16)
        d = CustomTorch.custom_relu(d)
        d = CustomTorch.custom_max_pool2d(d, kernel_size=2)

        d = CustomTorch.custom_conv2d(d, 16, 32, kernel_size=3, stride=1, padding=1)
        d = CustomTorch.custom_batch_norm2d(d, 32)
        d = CustomTorch.custom_relu(d)
        d = CustomTorch.custom_max_pool2d(d, kernel_size=2)
        
        d = CustomTorch.custom_conv2d(d, 32, 64, kernel_size=3, stride=1, padding=1)
        d = CustomTorch.custom_batch_norm2d(d, 64)
        d = CustomTorch.custom_relu(d)
        d = CustomTorch.custom_max_pool2d(d, kernel_size=2)

        d = d.view(d.size(0)-1)
        d = CustomTorch.custom_linear(d, 64 * 28 * 28, 256)
        d = CustomTorch.custom_relu(d)
        d = CustomTorch.custom_linear(256, num_classes)

        return d


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Add any additional data augmentation and preprocessing steps
])
torch.manual_seed(42)
learning_rate = 0.001
batch_size = 32
epoch_size = 10
class_size = 10


train_set = datasets.ImageFolder('./data/train_data', transform = data_transforms )
test_set = datasets.ImageFolder('./data/test_data', transform = data_transforms)

train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)


model = Plant_Identifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters, lr=learning_rate)


# training loop to train the model
for epoch in range(epoch_size):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        # perform backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print(f"Epoch: [{epoch+1}/{epoch_size}], Loss: {loss.item():.4f}")


