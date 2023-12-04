import torch
# TODO: implement the below custom functions
class CustomTorch:

    # each filter is a collection of kernels with there being 1 kernel for every single input channel and each kernel being unique
    # each filter in a convolutional layer produces 1 and only 1 output channel
    # Each per channel processed version are then summed together to form one channel and the filter as a whole produces one overall output channel
    def custom_conv2d(x, n_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        print(f"Conv2D parameters -- n_channels: {n_channels}, out_channels: {out_channels}, kernel_size: {kernel_size}, stride: {stride}, padding: {padding}, dilation: {dilation}, groups: {groups}, bias: {bias}, padding_mode: {padding_mode}, device: {device}, dtype: {dtype} ")
        batch_size, in_channels, in_height, in_width = x.shape
          # Initialize output tensor
        out_height = ((in_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)
        out_width = ((in_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)
        output = torch.zeros(batch_size, out_channels, out_height, out_width, device=device, dtype=dtype
                             
                             )

         # Perform convolution
        for b in range(batch_size):
            for oc in range(out_channels):
                for ic in range(n_channels):
                    for i in range(out_height):
                        for j in range(out_width):
                            for k in range(kernel_size):
                                for l in range(kernel_size):
                                    h = i * stride + k * dilation - padding
                                    w = j * stride + l * dilation - padding
                                    if (
                                        h >= 0
                                        and h < in_height
                                        and w >= 0
                                        and w < in_width
                                    ):
                                        output[b, oc, i, j] += (
                                            x[b, ic, h, w] * weights[oc, ic, k, l]
                                        )

                # Add bias if enabled
                if bias:
                    output[b, oc] += biases[oc]


        return


    # Normalization --> refers to the process of standardizing or rescaling the input data to have a zero mean and unit variance
    # the input tensor z is normalized using the mean and variance calculated within the current minibatch
    def custom_batch_norm2d(x, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        print(f"BatchNorm2D parameters -- num_features: {num_features}, eps: {eps}, momentum: {momentum}, affine: {affine}, track_running_stats: {track_running_stats}, device: {device}, dtype: {dtype}")
        # training the model based on the data given
        if track_running_stats:
            mean = mean(dim = (0, 2, 3))
            variance = variance(dim=(0, 2, 3), unbiased = False)

            # update the mean and variance
            running_mean = (1 - momentum) * running_mean + momentum * mean
            running_variance = (1-momentum) * running_variance + momentum * variance

            # normlize the input
            x_normalized = ( x- mean [None, :, None, None])/ torch.sqrt()
             # Scale and shift by learnable parameters
            out = weight[None, :, None, None] * x_normalized + bias[None, :, None, None]
            
        else:
            # Normalize the input tensor using running mean and variance
            x_normalized = (x - running_mean[None, :, None, None]) / torch.sqrt(
                running_var[None, :, None, None] + eps)

            # Scale and shift by learnable parameters
            out = weight[None, :, None, None] * x_normalized + bias[None, :, None, None]
            
        return out


    def custom_max_pool2d(x, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        print(f"MaxPool2D parameters -- kernel_size: {kernel_size}, stride: {stride}, padding: {padding}, dilation: {dilation}, return_indicies: {return_indices}, ceil_mode: {ceil_mode}")
        if stride is None:
            stride = kernel_size
        batch_size, channels, height, width = x.size()
        output_height = (height - kernel_size + 2 * padding) // stride + 1
        output_width = (width - kernel_size + 2 * padding) // stride + 1

        output = torch.zeros(batch_size, channels, output_height, output_width)

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * stride
                h_end = h_start + kernel_size
                w_start = j * stride
                w_end = w_start + kernel_size
                output[:,:,i,j] += torch.max(torch.max(x[:, :, h_start:h_end, w_start:w_end], dim=2)[0], dim=2)[0]
        return output


    def custom_relu(x, inplace=False):
        print(f"ReLU parameters -- inplace: {inplace}")
        output = torch.maximum(x, torch.tensor(0.0))
        return output

    def custom_linear(x, in_features, out_features):
        print('fsfs')
        return


