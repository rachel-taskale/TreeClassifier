import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


num_classes = 5

class CustomConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
            super(CustomConv2d, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.reset_parameters()
        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.bias)
        def forward(self, x):
            batch_size, in_channels, in_height, in_width = x.size()
            out_height = int(((in_height + 2 * self.padding - self.kernel_size) / self.stride) + 1)
            out_width = int(((in_width + 2 * self.padding - self.kernel_size) / self.stride) + 1)

            x_padded = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))
            out = torch.zeros((batch_size, self.out_channels, out_height, out_width), device=x.device)

            for i in range(out_height):
                for j in range(out_width):
                    x_window = x_padded[:, :, i * self.stride:i * self.stride + self.kernel_size,
                                        j * self.stride:j * self.stride + self.kernel_size]
                    out[:, :, i, j] = torch.sum(
                        x_window.unsqueeze(1) * self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), dim=(2, 3, 4)
                    ) + self.bias.unsqueeze(0)

            return out
                             
class CustomBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, track_running_stats=True, momentum=0.1):
        super(CustomBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.track_running_stats = True
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_variance', torch.ones(1, num_features, 1, 1))

    def forward(self, input):
        if self.training and self.track_running_stats:
            mean = input.mean(dim=(0, 2, 3), keepdim=True)
            variance = input.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_variance = (1 - self.momentum) * self.running_variance + self.momentum * variance

            input_normalized = (input - mean) / torch.sqrt(variance + self.eps).expand_as(input)
            out = self.weight * input_normalized + self.bias
        else:
            input_normalized = (input - self.running_mean) / torch.sqrt(self.running_variance + self.eps).expand_as(input)
            out = self.weight * input_normalized + self.bias

        return out

class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()
        return
    def forward(self, x, inplace=False):
        print('in relu')
        print(f"ReLU parameters -- inplace: {inplace}")
        output = torch.maximum(x, torch.tensor(0.0))
        return output


class CustomMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, celi_mode=False):
        super(CustomMaxPool2d, self).__init__()
        if stride is None:
            self.stride = kernel_size
        self.kernel_size = kernel_size
        self.padding = padding
    def forward(self, x):
        print('in max pool')
        batch_size, channels, height, width = x.size()
        output_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
        output = torch.zeros(batch_size, channels, output_height, output_width)

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                output[:,:,i,j] += torch.max(torch.max(x[:, :, h_start:h_end, w_start:w_end], dim=2)[0], dim=2)[0]
        return output





class Plant_Identifier(nn.Module):
    def __init__(self):
        super(Plant_Identifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        # pass

    def forward(self, d):
        d = self.conv1(d)
        d = self.batchnorm1(d)
        d = self.relu1(d)
        d = self.maxpool1(d)
        
        d = self.conv2(d)
        d = self.batchnorm2(d)
        d = self.relu2(d)
        d = self.maxpool2(d)
        
        d = self.conv3(d)
        d = self.batchnorm3(d)
        d = self.relu3(d)
        d = self.maxpool3(d)
        
        d = d.view(d.size(0), -1)
        d = self.fc1(d)
        d = self.relu4(d)
        d = self.fc2(d)
        return d



class Plant_Identifier_Custom(nn.Module):
    def __init__(self):
        super(Plant_Identifier_Custom, self).__init__()
        self.conv1 = nn.Conv2d(3,16, kernel_size=2, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.maxpool1 = CustomMaxPool2d( kernel_size=2)
        
        self.conv2 = nn.Conv2d( 16, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 =CustomReLU()
        self.maxpool2 = CustomMaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d( 32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 =nn.ReLU()
        self.maxpool3 = CustomMaxPool2d( kernel_size=2)
        
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.relu4 =nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, d):
        d = self.conv1(d)
        d = self.batchnorm1(d)
        d = self.relu1(d)
        d = self.maxpool1(d)
        
        d = self.conv2(d)
        d = self.batchnorm2(d)
        d = self.relu2(d)
        d = self.maxpool2(d)
        
        d = self.conv3(d)
        d = self.batchnorm3(d)
        d = self.relu3(d)
        d = self.maxpool3(d)
        
        d = d.view(d.size(0), -1)
        d = self.fc1(d)
        d = self.relu4(d)
        d = self.fc2(d)
        print('done')
        return d


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Add any additional data augmentation and preprocessing steps
])
torch.manual_seed(42)
learning_rate = 0.005
batch_size = 32
epoch_size = 30
class_size = 5

train_set = datasets.ImageFolder('./data/training_data/', transform = data_transforms )
test_set = datasets.ImageFolder('./data/test_data/', transform = data_transforms)

train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)

compare_model = Plant_Identifier()


def training_function(model, optimizer, criterion):
    print('in training')
    for epoch in range(epoch_size):
        for images, labels in train_loader:
            print(labels)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # perform backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f"Epoch: [{epoch+1}/{epoch_size}], Loss: {loss.item():.4f}")


def prediction(model):
    print('in prediction')
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    print(total_correct)
    print(total_samples)
    accuracy = 100 * total_correct / total_samples
    print(f'Test Accuracy: {accuracy:.2f}%')



# --------------------------------------------------------------------------------------


# custom_model = Plant_Identifier_Custom()

criterion = nn.CrossEntropyLoss()
# custom_optimizer = optim.SGD(custom_model.parameters(), lr=learning_rate)
compare_optimizer = optim.SGD(compare_model.parameters(), lr=learning_rate)

# training_function(custom_model, compare_optimizer, criterion)
training_function(compare_model, compare_optimizer, criterion)

# Evaluation
# custom_model.eval()
compare_model.eval()
total_correct = 0
total_samples = 0


# prediction(custom_model)
prediction(compare_model)







