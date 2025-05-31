class PoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.last_input = None
        self.max_indices = None

    def forward(self, input_batch):
        self.last_input = input_batch
        batch_size = len(input_batch)
        channels = len(input_batch[0])
        input_height = len(input_batch[0][0])
        input_width = len(input_batch[0][0][0])
        
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        
        output = []
        self.max_indices = []
        
        for b in range(batch_size):
            batch_output = []
            batch_indices = []
            
            for c in range(channels):
                channel_output = []
                channel_indices = []
                
                for i in range(output_height):
                    row_output = []
                    row_indices = []
                    
                    for j in range(output_width):
                        max_val = float('-inf')
                        max_i, max_j = 0, 0
                        
                        # Find maximum in pooling window
                        for pi in range(self.pool_size):
                            for pj in range(self.pool_size):
                                h_idx = i * self.stride + pi
                                w_idx = j * self.stride + pj
                                
                                if h_idx < input_height and w_idx < input_width:
                                    val = input_batch[b][c][h_idx][w_idx]
                                    if val > max_val:
                                        max_val = val
                                        max_i, max_j = h_idx, w_idx
                        
                        row_output.append(max_val)
                        row_indices.append((max_i, max_j))
                    
                    channel_output.append(row_output)
                    channel_indices.append(row_indices)
                
                batch_output.append(channel_output)
                batch_indices.append(channel_indices)
            
            output.append(batch_output)
            self.max_indices.append(batch_indices)
        
        return output

    def backward(self, output_gradient):
        # Placeholder - return zero gradients
        batch_size = len(self.last_input)
        channels = len(self.last_input[0])
        input_height = len(self.last_input[0][0])
        input_width = len(self.last_input[0][0][0])
        
        input_gradient = []
        for b in range(batch_size):
            batch_grad = []
            for c in range(channels):
                channel_grad = []
                for h in range(input_height):
                    row_grad = [0.0] * input_width
                    channel_grad.append(row_grad)
                batch_grad.append(channel_grad)
            input_gradient.append(batch_grad)
        
        return input_gradient