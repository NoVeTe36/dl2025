import time
import matplotlib.pyplot as plt

from layers.dense import DenseLayer
from layers.conv import ConvLayer
from layers.pooling import PoolingLayer
from optimizers.sgd import SGD
from loss.cross_entropy import CrossEntropy
from activations.relu import ReLU
from activations.softmax import Softmax


class CNN:
    def __init__(self):
        self.conv1 = ConvLayer(filters=6, kernel_size=5, input_channels=1)   # 28x28x1 -> 24x24x6
        self.pool1 = PoolingLayer(pool_size=2, stride=2)                    # 24x24x6 -> 12x12x6
        self.conv2 = ConvLayer(filters=16, kernel_size=5, input_channels=6)  # 12x12x6 -> 8x8x16
        self.pool2 = PoolingLayer(pool_size=2, stride=2)                    # 8x8x16 -> 4x4x16
        
        # Dense layers after convolution (4*4*16 = 256 features)
        self.dense1 = DenseLayer(256, 120)
        self.dense2 = DenseLayer(120, 84)
        self.dense3 = DenseLayer(84, 10)
        
        self.relu = ReLU()
        self.softmax = Softmax()
        self.loss_function = CrossEntropy()
        self.optimizer = SGD(learning_rate=0.01)
        
        # Store ALL layers for optimizer
        self.layers = [self.conv1, self.conv2, self.dense1, self.dense2, self.dense3]

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        
        # Convolutional layers
        z1 = self.conv1.forward(x)           # (batch, 6, 24, 24)
        a1 = self.relu.forward(z1)
        p1 = self.pool1.forward(a1)          # (batch, 6, 12, 12)
        
        z2 = self.conv2.forward(p1)          # (batch, 16, 8, 8)
        a2 = self.relu.forward(z2)
        p2 = self.pool2.forward(a2)          # (batch, 16, 4, 4)
        
        # flatten for dense layers
        flattened = []
        for sample in p2:  # p2 is (batch, 16, 4, 4)
            flat_sample = []
            for channel in sample:
                for row in channel:
                    flat_sample.extend(row)
            flattened.append(flat_sample)
        
        # Dense layers
        z3 = self.dense1.forward(flattened)   # (batch, 120)
        a3 = self.relu.forward(z3)
        
        z4 = self.dense2.forward(a3)          # (batch, 84)
        a4 = self.relu.forward(z4)
        
        z5 = self.dense3.forward(a4)          # (batch, 10)
        output = self.softmax.forward(z5)
        
        return output

    def backward(self, x, y):
        z1 = self.conv1.forward(x)
        a1 = self.relu.forward(z1)
        p1 = self.pool1.forward(a1)
        
        z2 = self.conv2.forward(p1)
        a2 = self.relu.forward(z2)
        p2 = self.pool2.forward(a2)
        
        flattened = []
        batch_size = len(p2)
        for sample in p2: 
            flat_sample = []
            for channel in sample:
                for row in channel:
                    flat_sample.extend(row)
            flattened.append(flat_sample)
            
        z3 = self.dense1.forward(flattened)
        a3 = self.relu.forward(z3)
        z4 = self.dense2.forward(a3)
        a4 = self.relu.forward(z4)
        z5 = self.dense3.forward(a4)
        predictions = self.softmax.forward(z5)
        
        loss = self.loss_function.calculate(predictions, y)
        grad = self.loss_function.gradient(predictions, y)
        grad = self.dense3.backward(grad)
        for b in range(len(grad)):
            for j in range(len(grad[b])):
                if z4[b][j] <= 0:
                    grad[b][j] = 0.0
        
        grad = self.dense2.backward(grad)
        for b in range(len(grad)):
            for j in range(len(grad[b])):
                if z3[b][j] <= 0:
                    grad[b][j] = 0.0
        
        grad = self.dense1.backward(grad)
        reshaped_grad = []
        for b in range(batch_size):
            sample_grad = []
            idx = 0
            for c in range(16):
                channel_grad = []
                for h in range(4):
                    row_grad = []
                    for w in range(4):
                        row_grad.append(grad[b][idx])
                        idx += 1
                    channel_grad.append(row_grad)
                sample_grad.append(channel_grad)
            reshaped_grad.append(sample_grad)
        

        grad = self.pool2.backward(reshaped_grad)
        for b in range(len(grad)):
            for f in range(len(grad[b])):
                for h in range(len(grad[b][f])):
                    for w in range(len(grad[b][f][h])):
                        if z2[b][f][h][w] <= 0:
                            grad[b][f][h][w] = 0.0
        
        grad = self.conv2.backward(grad)
        grad = self.pool1.backward(grad)
        for b in range(len(grad)):
            for f in range(len(grad[b])):
                for h in range(len(grad[b][f])):
                    for w in range(len(grad[b][f][h])):
                        if z1[b][f][h][w] <= 0:
                            grad[b][f][h][w] = 0.0
        
        grad = self.conv1.backward(grad)
        
        return loss

    def calculate_accuracy(self, data_loader, max_batches=None):
        """Calculate accuracy on dataset"""
        correct = 0
        total = 0
        batch_count = 0
        
        for batch_x, batch_y in data_loader.get_batches(batch_size=100):
            predictions = self.forward(batch_x)
            
            for i in range(len(predictions)):
                pred_class = predictions[i].index(max(predictions[i]))
                true_class = batch_y[i].index(max(batch_y[i]))
                if pred_class == true_class:
                    correct += 1
                total += 1
            
            batch_count += 1
            if max_batches and batch_count >= max_batches:
                break
        
        return correct / total if total > 0 else 0.0

    def train(self, data_loader, epochs, batch_size=32, learning_rate=None, early_stop_patience=5, early_stop_min_delta=0.001):
        """
        Train with early stopping and configurable learning rate
        
        Args:
            data_loader: DataLoader object
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for training (if None, uses optimizer's current rate)
            early_stop_patience: Number of epochs to wait for improvement
            early_stop_min_delta: Minimum change to qualify as improvement
        """
        
        if learning_rate is not None:
            print(f"Setting learning rate to: {learning_rate}")
            self.optimizer.learning_rate = learning_rate
        
        print(f"Using learning rate: {self.optimizer.learning_rate}")
        
        best_loss = float('inf')
        patience_counter = 0
        
        print(f"Training with early stopping (patience={early_stop_patience}, min_delta={early_stop_min_delta})")
        print("=" * 70)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            total_loss = 0.0
            batch_count = 0
            
            total_batches = (len(data_loader.images) + batch_size - 1) // batch_size
            print_interval = max(1, total_batches // 5)
            
            # Training phase
            for batch_x, batch_y in data_loader.get_batches(batch_size):
                loss = self.backward(batch_x, batch_y)
                self.optimizer.update(self.layers) # Update weights
                
                total_loss += loss
                batch_count += 1
                
                if batch_count % print_interval == 0 or batch_count == total_batches:
                    print(f"Epoch {epoch+1}, Batch {batch_count}/{total_batches}, Loss: {loss:.4f}")
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            epoch_time = time.time() - epoch_start_time
            accuracy = self.calculate_accuracy(data_loader, max_batches=5)
            
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)
            self.epoch_times.append(epoch_time)
            
            print(f"Epoch {epoch+1}/{epochs} completed:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Learning Rate: {self.optimizer.learning_rate}")
            print(f"  Time: {epoch_time:.2f}s")
            print("-" * 50)
            
            if avg_loss < best_loss - early_stop_min_delta:
                best_loss = avg_loss
                patience_counter = 0
                print(f"  ✓ New best loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{early_stop_patience}")
            
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                print(f"Best loss: {best_loss:.4f}")
                break
        
        print("\nTraining completed!")
        print(f"Total epochs: {len(self.train_losses)}")
        print(f"Final loss: {self.train_losses[-1]:.4f}")
        print(f"Final accuracy: {self.train_accuracies[-1]:.4f}")
        print(f"Final learning rate: {self.optimizer.learning_rate}")
        print(f"Total training time: {sum(self.epoch_times):.2f}s")
        
    def train_with_validation(self, train_loader, val_loader, epochs, batch_size=32, learning_rate=None, 
                            early_stop_patience=5, early_stop_min_delta=0.001):
        """
        Train with validation monitoring and early stopping
        """
        
        if not hasattr(self, 'train_losses'):
            self.train_losses = []
        if not hasattr(self, 'val_losses'):
            self.val_losses = []
        if not hasattr(self, 'train_accuracies'):
            self.train_accuracies = []
        if not hasattr(self, 'val_accuracies'):
            self.val_accuracies = []
        if not hasattr(self, 'epoch_times'):
            self.epoch_times = []
        
        if learning_rate is not None:
            print(f"Setting learning rate to: {learning_rate}")
            self.optimizer.learning_rate = learning_rate
        
        print(f"Using learning rate: {self.optimizer.learning_rate}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.val_losses = []
        self.val_accuracies = []
        
        print(f"Training with validation monitoring")
        print(f"Early stopping: patience={early_stop_patience}, min_delta={early_stop_min_delta}")
        print("=" * 70)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # TRAINING PHASE
            total_train_loss = 0.0
            train_batch_count = 0
            
            for batch_x, batch_y in train_loader.get_batches(batch_size):
                loss = self.backward(batch_x, batch_y)
                self.optimizer.update(self.layers)
                
                total_train_loss += loss
                train_batch_count += 1
                
                if train_batch_count % 10 == 0:
                    print(f"Epoch {epoch+1}, Training Batch {train_batch_count}, Loss: {loss:.4f}")
            
            # Validation phase
            total_val_loss = 0.0
            val_batch_count = 0
            val_correct = 0
            val_total = 0
            
            for batch_x, batch_y in val_loader.get_batches(batch_size):
                predictions = self.forward(batch_x)
                val_loss = self.loss_function.calculate(predictions, batch_y)
                
                total_val_loss += val_loss
                val_batch_count += 1
                
                for i in range(len(predictions)):
                    pred_class = predictions[i].index(max(predictions[i]))
                    true_class = batch_y[i].index(max(batch_y[i]))
                    if pred_class == true_class:
                        val_correct += 1
                    val_total += 1
                    
            avg_train_loss = total_train_loss / train_batch_count if train_batch_count > 0 else 0
            avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else 0
            val_accuracy = val_correct / val_total if val_total > 0 else 0
            epoch_time = time.time() - epoch_start_time
            
            train_accuracy = self.calculate_accuracy_subset(train_loader, max_batches=3)
            
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_accuracy)
            self.epoch_times.append(epoch_time)
            
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}   | Val Acc: {val_accuracy:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print("-" * 50)
            
            # Early stopping check 
            if avg_val_loss < best_val_loss - early_stop_min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{early_stop_patience}")
            
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
        
        print("\nTraining completed!")
        print(f"Total epochs: {len(self.train_losses)}")
        print(f"Final train loss: {self.train_losses[-1]:.4f}")
        print(f"Final val loss: {self.val_losses[-1]:.4f}")
        print(f"Best val loss: {min(self.val_losses):.4f}")

    def calculate_accuracy_subset(self, data_loader, max_batches=3):
        """Calculate accuracy on a subset of data for speed"""
        correct = 0
        total = 0
        batch_count = 0
        
        for batch_x, batch_y in data_loader.get_batches(batch_size=50):
            predictions = self.forward(batch_x)
            
            for i in range(len(predictions)):
                pred_class = predictions[i].index(max(predictions[i]))
                true_class = batch_y[i].index(max(batch_y[i]))
                if pred_class == true_class:
                    correct += 1
                total += 1
            
            batch_count += 1
            if batch_count >= max_batches:
                break
        
        return correct / total if total > 0 else 0.0

    def evaluate_dataset(self, data_loader):
        """Evaluate model on a complete dataset"""
        correct = 0
        total = 0
        
        for batch_x, batch_y in data_loader.get_batches(batch_size=100):
            predictions = self.forward(batch_x)
            
            for i in range(len(predictions)):
                pred_class = predictions[i].index(max(predictions[i]))
                true_class = batch_y[i].index(max(batch_y[i]))
                if pred_class == true_class:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy

    def plot_training_history(self, save_path=None):
        """Plot training history including loss, accuracy, and training time"""
        try:
            import matplotlib.pyplot as plt
            
            if not hasattr(self, 'train_losses') or not self.train_losses:
                print("No training history to plot")
                return
                
            epochs = range(1, len(self.train_losses) + 1)
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
            
            # Plot Loss
            ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
            if hasattr(self, 'val_losses') and self.val_losses:
                ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
            
            ax1.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot Accuracy
            if hasattr(self, 'train_accuracies') and self.train_accuracies:
                ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
            if hasattr(self, 'val_accuracies') and self.val_accuracies:
                ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
                
            ax2.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            # Plot Training Time per Epoch
            if hasattr(self, 'epoch_times') and self.epoch_times:
                ax3.plot(epochs, self.epoch_times, 'g-', label='Training Time per Epoch', linewidth=2, marker='o')
                
                ax3.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Time (seconds)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                if len(self.epoch_times) > 1:
                    avg_time = sum(self.epoch_times) / len(self.epoch_times)
                    ax3.axhline(y=avg_time, color='orange', linestyle='--', alpha=0.7, 
                            label=f'Average: {avg_time:.2f}s')
                    ax3.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Training history plot saved as: {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating plot: {e}")

    def evaluate(self, data_loader):
        """Evaluate model on full dataset"""
        print("Evaluating model on full dataset...")
        start_time = time.time()
        
        accuracy = self.calculate_accuracy(data_loader)
        eval_time = time.time() - start_time
        
        print(f"Final Accuracy: {accuracy:.4f}")
        print(f"Evaluation time: {eval_time:.2f}s")
        
        return accuracy