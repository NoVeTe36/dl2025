import os

def count_mnist_data(directory):
    train_images_path = os.path.join(directory, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(directory, 'train-labels.idx1-ubyte')
    
    if not os.path.exists(train_images_path) or not os.path.exists(train_labels_path):
        raise FileNotFoundError("MNIST data files not found in the specified directory.")
    
    with open(train_images_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')  # THIS is where number of images is

    with open(train_labels_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')  # Number of labels is here too

    if num_images != num_labels:
        raise ValueError(f"Number of images ({num_images}) and labels ({num_labels}) do not match.")
    
    return num_images

# plot a sample of the MNIST data
def plot_mnist_sample(directory, num_samples=10):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    train_images_path = os.path.join(directory, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(directory, 'train-labels.idx1-ubyte')

    if not os.path.exists(train_images_path) or not os.path.exists(train_labels_path):
        raise FileNotFoundError("MNIST data files not found in the specified directory.")

    with open(train_images_path, 'rb') as f:
        f.read(16)  # Skip header
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

    with open(train_labels_path, 'rb') as f:
        f.read(8)  # Skip header
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    directory = 'data'
    try:
        num_data = count_mnist_data(directory)
        plot_mnist_sample(directory, num_samples=10)
        print(f"Number of MNIST data entries in '{directory}': {num_data}")
    except Exception as e:
        print(f"Error: {e}")
