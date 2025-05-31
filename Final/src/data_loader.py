import struct
import os
from utils.random import CustomRandom

class DataLoader:
    def __init__(self, data_path, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, shuffle=True, random_seed=42, max_samples=None):
        if abs(train_ratio + test_ratio + val_ratio - 1.0) > 1e-6:
            raise ValueError("Not correct split ratios")
        
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        
        all_images, all_labels = self.load_mnist_data(data_path, max_samples)
        
        if random_seed is not None:
            random = CustomRandom(random_seed)
        
        if shuffle:
            combined = list(zip(all_images, all_labels))
            random.shuffle(combined)
            all_images, all_labels = zip(*combined)
            all_images, all_labels = list(all_images), list(all_labels)
        
        total_samples = len(all_images)
        train_end = int(total_samples * train_ratio)
        test_end = train_end + int(total_samples * test_ratio)
        
        self.train_images = all_images[:train_end]
        self.train_labels = all_labels[:train_end]
        
        self.test_images = all_images[train_end:test_end]
        self.test_labels = all_labels[train_end:test_end]
        
        self.val_images = all_images[test_end:]
        self.val_labels = all_labels[test_end:]
        
        self.images = self.train_images
        self.labels = self.train_labels
        
        print(f"Dataset split:")
        print(f"  Training: {len(self.train_images)} samples ({len(self.train_images)/total_samples*100:.1f}%)")
        print(f"  Testing: {len(self.test_images)} samples ({len(self.test_images)/total_samples*100:.1f}%)")
        print(f"  Validation: {len(self.val_images)} samples ({len(self.val_images)/total_samples*100:.1f}%)")

    def load_mnist_data(self, data_path, max_samples=None):
        images_path = os.path.join(data_path, 'train-images.idx3-ubyte')
        labels_path = os.path.join(data_path, 'train-labels.idx1-ubyte')
        
        # Load images
        with open(images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            
            # Use all images if max_samples is None, otherwise use the specified limit
            samples_to_load = num_images if max_samples is None else min(max_samples, num_images)
            print(f"Loading {samples_to_load} images from {num_images} total images...")
            
            images = []
            for i in range(samples_to_load):
                if i % 10000 == 0:
                    print(f"Loaded {i}/{samples_to_load} images...")
                
                image_data = []
                for _ in range(rows):
                    row = []
                    for _ in range(cols):
                        pixel = struct.unpack('B', f.read(1))[0]
                        row.append(pixel / 255.0)
                    image_data.append(row)
                images.append([image_data])

        with open(labels_path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            print(f"Loading {samples_to_load} labels from {num_labels} total labels...")
            
            labels = []
            for i in range(samples_to_load):
                label = struct.unpack('B', f.read(1))[0]
                one_hot = [0.0] * 10
                one_hot[label] = 1.0
                labels.append(one_hot)

        print(f"Successfully loaded {len(images)} images and {len(labels)} labels")
        return images, labels