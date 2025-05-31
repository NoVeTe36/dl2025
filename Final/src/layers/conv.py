from tensor import Tensor

class ConvLayer:
    def __init__(self, filters, kernel_size, stride=1, padding=0, input_channels=1):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_channels = input_channels
        
        # Initialize weights (filters, input_channels, kernel_size, kernel_size)
        self.weights = Tensor.random((filters, input_channels, kernel_size, kernel_size), 0.1)
        self.biases = Tensor.zeros(filters)
        
        # Store gradients
        self.grad_weights = None
        self.grad_biases = None
        
        # Store input for backward
        self.last_input = None

    def forward(self, input_batch):
        self.last_input = input_batch
        batch_size = len(input_batch)
        input_channels = len(input_batch[0])
        input_height = len(input_batch[0][0])
        input_width = len(input_batch[0][0][0])
        
        output_height = (input_height - self.kernel_size) // self.stride + 1
        output_width = (input_width - self.kernel_size) // self.stride + 1
        
        output = []
        for b in range(batch_size):
            batch_output = []
            for f in range(self.filters):
                filter_output = []
                for h in range(output_height):
                    row_output = []
                    for w in range(output_width):
                        conv_sum = self.biases.data[f]
                        
                        for c in range(input_channels):
                            for kh in range(self.kernel_size):
                                for kw in range(self.kernel_size):
                                    input_h = h * self.stride + kh
                                    input_w = w * self.stride + kw
                                    
                                    if (input_h < input_height and input_w < input_width):
                                        conv_sum += (input_batch[b][c][input_h][input_w] * 
                                                   self.weights.data[f][c][kh][kw])
                        
                        row_output.append(conv_sum)
                    filter_output.append(row_output)
                batch_output.append(filter_output)
            output.append(batch_output)
        
        return output

    def backward(self, output_gradient):
        batch_size = len(self.last_input)
        input_channels = len(self.last_input[0])
        input_height = len(self.last_input[0][0])
        input_width = len(self.last_input[0][0][0])
        
        input_gradient = []
        for b in range(batch_size):
            batch_grad = []
            for c in range(input_channels):
                channel_grad = []
                for h in range(input_height):
                    row_grad = [0.0] * input_width
                    channel_grad.append(row_grad)
                batch_grad.append(channel_grad)
            input_gradient.append(batch_grad)
        
        return input_gradient