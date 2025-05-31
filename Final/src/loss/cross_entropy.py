import math

class CrossEntropy:
    def calculate(self, predictions, targets):
        batch_size = len(predictions)
        total_loss = 0.0
        
        for i in range(batch_size):
            sample_loss = 0.0
            for j in range(len(predictions[i])):
                if targets[i][j] > 0:
                    pred_clipped = max(1e-15, min(1.0 - 1e-15, predictions[i][j]))
                    sample_loss += -targets[i][j] * math.log(pred_clipped)
            total_loss += sample_loss
        
        return total_loss / batch_size

    def gradient(self, predictions, targets):
        batch_size = len(predictions)
        num_classes = len(predictions[0])
        
        gradients = []
        for i in range(batch_size):
            sample_grad = []
            for j in range(num_classes):
                sample_grad.append(predictions[i][j] - targets[i][j])
            gradients.append(sample_grad)
        
        return gradients