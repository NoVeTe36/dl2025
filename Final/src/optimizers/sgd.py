class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_tensor(self, weights, gradients, indices=[]):
        """Recursively update weights of any dimension"""
        if isinstance(weights[0], list):
            for i in range(len(weights)):
                self.update_tensor(weights[i], gradients[i], indices + [i])
        else:
            for i in range(len(weights)):
                weights[i] -= self.learning_rate * gradients[i]

    def update(self, layers):
        for layer in layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'grad_weights'):
                if layer.grad_weights is not None:
                    self.update_tensor(layer.weights.data, layer.grad_weights.data)
                    
                    if hasattr(layer, 'biases') and hasattr(layer, 'grad_biases'):
                        if layer.grad_biases is not None:
                            for i in range(len(layer.biases.data)):
                                layer.biases.data[i] -= self.learning_rate * layer.grad_biases.data[i]