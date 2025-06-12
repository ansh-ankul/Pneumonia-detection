import os
import argparse
from models.cnn_model import create_model, train_model, save_model
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.suptitle('Training History', fontsize=16)
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('training_history.png')
    plt.close()
    print('Training history plot saved as training_history.png')

def main():
    parser = argparse.ArgumentParser(description='Train Pneumonia Detection Model')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation data directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--model_save_path', type=str, default='models/pneumonia_model.pth', help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Create model
    model = create_model()
    
    # Train model
    history, model = train_model(
        model,
        args.train_dir,
        args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    save_model(model, args.model_save_path)
    
    # Plot training history
    plot_training_history(history)

if __name__ == '__main__':
    main() 