import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
import os
from sklearn.utils.class_weight import compute_class_weight

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # Block 1
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 8, kernel_size=3, padding=1),  # Block 2
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 8, kernel_size=3, padding=1),  # Block 3
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # Block 4
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # Block 5
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Output size after 5 poolings from 150x150: 150/32 = 4.6875 -> 4
        # So, 64 x 4 x 4 = 1024
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def create_model():
    model = PneumoniaCNN()
    return model

def train_model(model, train_dir, validation_dir, epochs=20, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    import copy
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    # Data augmentation and normalization (to match Keras)
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, shear=0.2, scale=(0.8, 1.2)),  # shear and zoom
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.0], [1.0])  # grayscale, rescale=1./255
    ])

    val_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.0], [1.0])  # grayscale, only rescale
    ])

    # Create datasets (force grayscale)
    train_dataset = datasets.ImageFolder(train_dir, transform=lambda img: train_transform(img.convert('L')))
    val_dataset = datasets.ImageFolder(validation_dir, transform=lambda img: val_transform(img.convert('L')))

    # Compute class weights
    labels = [label for _, label in train_dataset.imgs]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss(weight=class_weights[1]) if len(class_weights) == 2 else nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00015, weight_decay=1e-4)
    # ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True, min_lr=1e-6)

    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            inputs = inputs.view(inputs.size(0), 1, 150, 150)  # Ensure shape [B, 1, 150, 150]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                inputs = inputs.view(inputs.size(0), 1, 150, 150)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                
                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # ReduceLROnPlateau step
        scheduler.step(val_loss)
        
        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f'Best model saved at epoch {best_epoch} with val_loss {best_val_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}. Best val_loss: {best_val_loss:.4f} at epoch {best_epoch}')
                model.load_state_dict(best_model_wts)
                break
    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    return history, model

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def load_model(model_path):
    model = PneumoniaCNN()
    model.load_state_dict(torch.load(model_path))
    return model

def predict_image(model, image_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.0], [1.0])
    ])
    
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(image)
    
    return prediction.item() 