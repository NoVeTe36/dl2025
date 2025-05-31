class ReLU:
    def __init__(self):
        self.last_input = None
    
    def forward(self, input_data):
        self.last_input = input_data
        
        def apply_relu(data):
            if isinstance(data, list):
                return [apply_relu(item) for item in data]
            else:
                return max(0.0, data)
        
        return apply_relu(input_data)
    
    def backward(self, output_gradient):
        def apply_relu_derivative(grad, inp):
            if isinstance(grad, list) and isinstance(inp, list):
                if len(grad) != len(inp):
                    raise ValueError(f"Gradient and input dimensions don't match")
                return [apply_relu_derivative(g, i) for g, i in zip(grad, inp)]
            else:
                return grad if inp > 0 else 0.0
        
        if self.last_input is None:
            raise ValueError("No input stored for backward")
        
        return apply_relu_derivative(output_gradient, self.last_input)