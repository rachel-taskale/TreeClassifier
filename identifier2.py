import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


num_classes = 5




class Plant_Identifier_Custom(nn.Module):
    def __init__(self):
        super(Plant_Identifier_Custom, self).__init__()
        
        self.conv1 = self.custom_conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = self.custom_conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = self.custom_conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.maxpool = self.custom_maxpool2d(kernel_size=2, stride=2)

        self.fc = torch.nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.custom_relu(self.conv1(x))
        x = self.maxpool(x)

        x = self.custom_relu(self.conv2(x))
        x = self.maxpool(x)

        x = self.custom_relu(self.conv3(x))
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        print('done')
        return x
    

    def custom_conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        bias = torch.nn.Parameter(torch.Tensor(out_channels))
        torch.nn.init.xavier_uniform_(weight)
        torch.nn.init.constant_(bias, 0)
        
        def conv2d_fn(self, x):
            B, C, H, W = x.size()
            H_out = (H + 2 * padding - kernel_size) // stride + 1
            W_out = (W + 2 * padding - kernel_size) // stride + 1
            output = torch.zeros(B, out_channels, H_out, W_out)
            
            padded_input = torch.nn.functional.pad(x, (padding, padding, padding, padding))

            for batch_idx in range(B):
                for out_channel_idx in range(out_channels):
                    for in_channel_idx in range(in_channels):
                        for h_idx in range(H_out):
                            for w_idx in range(W_out):
                                h_start = h_idx * stride
                                h_end = h_start + kernel_size
                                w_start = w_idx * stride
                                w_end = w_start + kernel_size
                                receptive_field = padded_input[batch_idx, in_channel_idx, h_start:h_end, w_start:w_end]
                                output[batch_idx, out_channel_idx, h_idx, w_idx] += torch.sum(receptive_field * weight[out_channel_idx, in_channel_idx]) + bias[out_channel_idx]
            
            return output
        return conv2d_fn
                                
    def custom_batchnorm2d(self, num_features, eps=1e-5, track_running_stats=True, momentum=0.1):
        def batchnorm2d_fn(self, x):
            if self.training and self.track_running_stats:
                mean = x.mean(dim=(0, 2, 3), keepdim=True)
                variance = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_variance = (1 - self.momentum) * self.running_variance + self.momentum * variance

                x_normalized = (x - mean) / torch.sqrt(variance + self.eps).expand_as(x)
                out = self.weight * x_normalized + self.bias
            else:
                x_normalized = (x - self.running_mean) / torch.sqrt(self.running_variance + self.eps).expand_as(x)
                out = self.weight * x_normalized + self.bias
            return out
        return batchnorm2d_fn

    def custom_relu(self):
        def relu_fn(self, x):
            print('in relu')
            output = torch.maximum(x, torch.tensor(0.0))
            return output
        return relu_fn


    def custom_maxpool2d(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, celi_mode=False):

        if stride is None:
            self.stride = kernel_size

        def maxpool_fn(self, x):
            B, C, H, W = x.size()
            H_out = (H - kernel_size) // stride + 1
            W_out = (W - kernel_size) // stride + 1
            output = torch.zeros(B, C, H_out, W_out)

            for batch_idx in range(B):
                for channel_idx in range(C):
                    for h_idx in range(H_out):
                        for w_idx in range(W_out):
                            h_start = h_idx * stride
                            h_end = h_start + kernel_size
                            w_start = w_idx * stride
                            w_end = w_start + kernel_size
                            receptive_field = x[batch_idx, channel_idx, h_start:h_end, w_start:w_end]
                            output[batch_idx, channel_idx, h_idx, w_idx] = torch.max(receptive_field)

            return output
        return maxpool_fn

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


custom_model = Plant_Identifier_Custom()

criterion = nn.CrossEntropyLoss()
custom_optimizer = optim.SGD(custom_model.parameters(), lr=learning_rate)

training_function(custom_model, custom_optimizer, criterion)

# Evaluation
custom_model.eval()
# compare_model.eval()
total_correct = 0
total_samples = 0


prediction(custom_model)
# prediction(compare_model)







