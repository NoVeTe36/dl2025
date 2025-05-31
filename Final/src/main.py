from data_loader import DataLoader
from network import CNN

from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

class DataLoaderWrapper:
    """Wrapper to make split datasets compatible with existing train method"""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def get_batches(self, batch_size=32):
        for i in range(0, len(self.images), batch_size):
            batch_images = self.images[i:i+batch_size]
            batch_labels = self.labels[i:i+batch_size]
            yield batch_images, batch_labels

def main():
    print("=== CNN with Train/Test/Validation Split ===")
    print("Loading MNIST data...")
    data_loader = DataLoader(
        'data', 
        train_ratio=0.8,
        test_ratio=0.1, 
        val_ratio=0.1,  
        shuffle=True,
        random_seed=42,
        max_samples=1000
    )
    
    print("Creating CNN model...")
    model = CNN()
    
    model.train_losses = []
    model.val_losses = []
    model.train_accuracies = []
    model.val_accuracies = []
    model.epoch_times = []
    
    model.train_losses = []
    model.train_accuracies = []
    model.epoch_times = []
    
    print("\nStarting training...")
    train_loader = DataLoaderWrapper(data_loader.train_images, data_loader.train_labels)
    val_loader = DataLoaderWrapper(data_loader.val_images, data_loader.val_labels)
    
    model.train_with_validation(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.getint('training', 'epochs'), 
        batch_size=config.getint('training', 'batch_size'),
        learning_rate=config.getfloat('training', 'learning_rate'),
        early_stop_patience=config.getint('training', 'early_stop_patience'),
        early_stop_min_delta=config.getfloat('training', 'early_stop_min_delta')
    )
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nEvaluating on Training Set...")
    train_accuracy = model.evaluate_dataset(train_loader)
    
    print("\nEvaluating on Validation Set...")
    val_accuracy = model.evaluate_dataset(val_loader)
    
    print("\nEvaluating on Test Set...")
    test_loader = DataLoaderWrapper(data_loader.test_images, data_loader.test_labels)
    test_accuracy = model.evaluate_dataset(test_loader)
    
    print("\nPlotting training history...")
    try:
        model.plot_training_history(save_path='training_history_cnn.png')
    except Exception as e:
        print(f"Error plotting: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Architecture: Real CNN with Convolution + Pooling + Dense layers")
    print(f"Dataset splits:")
    print(f"  Training samples: {len(data_loader.train_images)}")
    print(f"  Validation samples: {len(data_loader.val_images)}")
    print(f"  Test samples: {len(data_loader.test_images)}")
    print(f"\nModel Performance:")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")  
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    if hasattr(model, 'train_losses') and model.train_losses:
        print(f"\nTraining Stats:")
        print(f"  Total epochs trained: {len(model.train_losses)}")
        print(f"  Final training loss: {model.train_losses[-1]:.4f}")
        print(f"  Final training accuracy: {model.train_accuracies[-1]:.4f}")
        if hasattr(model, 'val_losses') and model.val_losses:
            print(f"  Final validation loss: {model.val_losses[-1]:.4f}")
            print(f"  Final validation accuracy: {model.val_accuracies[-1]:.4f}")
            print(f"  Best validation accuracy: {max(model.val_accuracies):.4f}")
        print(f"  Average time per epoch: {sum(model.epoch_times)/len(model.epoch_times):.2f}s")
        print(f"  Total training time: {sum(model.epoch_times):.2f}s")

        
        # Check for overfitting
        if train_accuracy - val_accuracy > 0.1:
            print(f"\nWarning: Possible overfitting detected!")
            print(f"   Training accuracy ({train_accuracy:.4f}) >> Validation accuracy ({val_accuracy:.4f})")
        elif abs(train_accuracy - val_accuracy) < 0.05:
            print(f"\nFine: Training and validation accuracies are close")

        if len(model.train_losses) > 1:
            loss_improvement = model.train_losses[0] - model.train_losses[-1]
            print(f"\nTraining Progress:")
            print(f"  Loss improvement: {loss_improvement:.4f}")
            if hasattr(model, 'val_accuracies') and model.val_accuracies:
                print(f"  Best validation accuracy: {max(model.val_accuracies):.4f}")

if __name__ == "__main__":
    main()