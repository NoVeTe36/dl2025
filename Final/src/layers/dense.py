from tensor import Tensor

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights = Tensor.random((input_size, output_size), 0.1)
        self.biases = Tensor.zeros(output_size)
        
        # Store gradients
        self.grad_weights = None
        self.grad_biases = None
        
        self.last_input = None

    def forward(self, input_batch):
        self.last_input = input_batch
        batch_size = len(input_batch)
        if batch_size == 0:
            raise ValueError("Input batch cannot be empty.")
        
        if isinstance(input_batch[0], list) and isinstance(input_batch[0][0], list):
            flattened_input = []
            for sample in input_batch:
                flat_sample = []
                for channel in sample:
                    for row in channel:
                        flat_sample.extend(row)
                flattened_input.append(flat_sample)
            input_batch = flattened_input

        output = []
        for sample in input_batch:
            sample_output = []
            for j in range(self.output_size):
                value = self.biases.data[j]
                for i in range(self.input_size):
                    value += sample[i] * self.weights.data[i][j]
                sample_output.append(value)
            output.append(sample_output)
        
        return output

    def backward(self, output_gradient):
        batch_size = len(output_gradient)
        if batch_size == 0:
            raise ValueError("Output gradient cannot be empty.")
        
        if batch_size > 0:
            if len(output_gradient[0]) != self.output_size:
                raise ValueError(f"Output gradient size {len(output_gradient[0])} doesn't match layer output size {self.output_size}")
        
        if isinstance(self.last_input[0], list) and isinstance(self.last_input[0][0], list):
            flattened_input = []
            for sample in self.last_input:
                flat_sample = []
                for channel in sample:
                    for row in channel:
                        flat_sample.extend(row)
                flattened_input.append(flat_sample)
            input_data = flattened_input
        else:
            input_data = self.last_input

        self.grad_weights = Tensor.zeros((self.input_size, self.output_size))
        self.grad_biases = Tensor.zeros(self.output_size)
        
        input_gradient = [[0.0] * self.input_size for _ in range(batch_size)]
        
        for b in range(batch_size):
            for j in range(self.output_size):
                grad_out = output_gradient[b][j]
                self.grad_biases.data[j] += grad_out
                                
                for i in range(self.input_size):                    
                    self.grad_weights.data[i][j] += input_data[b][i] * grad_out                    
                    input_gradient[b][i] += self.weights.data[i][j] * grad_out
        
        for j in range(self.output_size):
            self.grad_biases.data[j] /= batch_size
            for i in range(self.input_size):
                self.grad_weights.data[i][j] /= batch_size
        
        return input_gradient