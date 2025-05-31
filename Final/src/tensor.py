from utils.random import CustomRandom

class Tensor:
    def __init__(self, data):
        self.data = data
        self.grad = None

    @staticmethod
    def random(shape, scale=0.1):
        rng = CustomRandom()
        
        if len(shape) == 4:  # 4D tensor
            data = []
            for i in range(shape[0]):
                channels = []
                for j in range(shape[1]):
                    rows = []
                    for k in range(shape[2]):
                        row = []
                        for l in range(shape[3]):
                            row.append(rng.uniform(-scale, scale))
                        rows.append(row)
                    channels.append(rows)
                data.append(channels)
            return Tensor(data)
        
        elif len(shape) == 2:  # 2D tensor
            data = []
            for i in range(shape[0]):
                row = []
                for j in range(shape[1]):
                    row.append(rng.uniform(-scale, scale))
                data.append(row)
            return Tensor(data)
        
        else:  # 1D tensor
            data = [rng.uniform(-scale, scale) for _ in range(shape[0])]
            return Tensor(data)

    @staticmethod
    def zeros(shape):
        if isinstance(shape, int):
            return Tensor([0.0] * shape)
        
        if len(shape) == 1:
            return Tensor([0.0] * shape[0])
        
        elif len(shape) == 2:
            data = []
            for i in range(shape[0]):
                data.append([0.0] * shape[1])
            return Tensor(data)
        
        elif len(shape) == 4:
            data = []
            for i in range(shape[0]):
                channels = []
                for j in range(shape[1]):
                    rows = []
                    for k in range(shape[2]):
                        rows.append([0.0] * shape[3])
                    channels.append(rows)
                data.append(channels)
            return Tensor(data)
        
        return Tensor([])

    def shape(self):
        def get_dims(data):
            if not isinstance(data, list):
                return []
            if len(data) == 0:
                return [0]
            return [len(data)] + get_dims(data[0])
        return get_dims(self.data)

    def flatten(self):
        result = []
        
        def add_items(data):
            if isinstance(data, list):
                for item in data:
                    add_items(item)
            else:
                result.append(data)
        
        add_items(self.data)
        return result

    def mean(self):
        flat = self.flatten()
        if len(flat) == 0:
            return 0.0
        return sum(flat) / len(flat)