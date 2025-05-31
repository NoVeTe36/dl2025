class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layers):
        for layer in layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'grad_weights'):
                if layer.grad_weights is not None:
                    # Update weights
                    for i in range(len(layer.weights.data)):
                        for j in range(len(layer.weights.data[i])):
                            layer.weights.data[i][j] -= self.learning_rate * layer.grad_weights.data[i][j]
                    
                    # Update biases
                    if hasattr(layer, 'biases') and hasattr(layer, 'grad_biases'):
                        if layer.grad_biases is not None:
                            for i in range(len(layer.biases.data)):
                                layer.biases.data[i] -= self.learning_rate * layer.grad_biases.data[i]