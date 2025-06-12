import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from models.cnn_model import PneumoniaCNN, load_model
from PIL import Image

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
TEST_DIR = 'data/chest_xray/test'
MODEL_PATH = 'models/pneumonia_model.pth'

# Data transform (must match training)
test_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.0], [1.0])
])

def get_test_loader():
    dataset = datasets.ImageFolder(TEST_DIR, transform=lambda img: test_transform(img.convert('L')))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return loader, dataset.classes

def evaluate_model():
    loader, class_names = get_test_loader()
    model = PneumoniaCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            inputs = inputs.view(inputs.size(0), 1, 150, 150)
            outputs = model(inputs)
            preds = (outputs.squeeze() > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {acc*100:.2f}%')
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print('Confusion matrix saved as confusion_matrix.png')

    return loader, class_names, all_preds, all_labels

def visualize_predictions(loader, class_names, num_images=12):
    # Get a batch of test images
    data_iter = iter(loader)
    images, labels = next(data_iter)
    images = images[:num_images]
    labels = labels[:num_images]

    model = PneumoniaCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = images.to(device)
        inputs = inputs.view(inputs.size(0), 1, 150, 150)
        outputs = model(inputs)
        preds = (outputs.squeeze() > 0.5).long().cpu().numpy()

    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        img = images[i].squeeze().numpy()
        plt.subplot(2, num_images//2, i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10)
    plt.tight_layout()
    plt.savefig('test_predictions.png')
    plt.close()
    print('Test predictions saved as test_predictions.png')

if __name__ == '__main__':
    loader, class_names, all_preds, all_labels = evaluate_model()
    visualize_predictions(loader, class_names) 