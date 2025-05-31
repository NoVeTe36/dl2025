class CustomRandom:
    def __init__(self, seed=123456789):
        self.seed = seed
        self.mult = 16807
        self.mod = (2**31) - 1
        self.current = seed
    
    def random(self):
        self.current = (self.mult * self.current + 1) % self.mod
        return self.current / self.mod
    
    def random_list(self, size):
        return [self.random() for _ in range(size)]
    
    def uniform(self, low, high):
        return low + (high - low) * self.random()
    
    def shuffle(self, data):
        shuffled_data = data[:]
        for i in range(len(shuffled_data) - 1, 0, -1):
            j = int(self.random() * (i + 1))
            shuffled_data[i], shuffled_data[j] = shuffled_data[j], shuffled_data[i]
        return shuffled_data
