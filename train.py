import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from MedMamba import VSSM as medmamba


def safe_divide(numerator, denominator):
    """Safe division that returns 0 if denominator is 0"""
    return numerator / denominator if denominator != 0 else 0.0

def plot_metrics(epochs, train_metrics, val_metrics, metric_name):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_metrics, label=f'Training {metric_name}')
    plt.plot(range(1, epochs+1), val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    plt.grid(True)
    pth = '/content/drive/MyDrive/mamba/Training and Validation '+metric_name
    plt.savefig(pth)
    #plt.show()

def plot_confusion_matrix(true_labels, predicted_labels, classes, title='Confusion Matrix'):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    pth = '/content/drive/MyDrive/mamba/Confusion matrix'
    plt.savefig(pth)
    #plt.show()

def main():
    # Configuration
    train=True
    use_prev_trained = False
    save_every_epoch = False
    data_dir = "/content/drive/MyDrive/mamba/chest_xray"
    batch_size = 8
    epochs = 100
    learning_rate = 0.0001
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    # Data transformations
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(312),
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.3, 0.3, 0.3])
        ]),
        # "val": transforms.Compose([
        #     transforms.Resize(312),
        #     transforms.CenterCrop(256),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.20, 0.20, 0.20])
        # ]),
        "test": transforms.Compose([
            transforms.Resize(312),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.3, 0.3, 0.3])
        ])
    }

    # Load datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"),
                                       #transform=data_transform["train"]
                                       )
    # val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"),
    #                                  transform=data_transform["val"])
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"),
                                      #transform=data_transform["test"]
                                      )
    #train_dataset = train_dataset.make_dataset(test_dataset)

    print("Train length:",len(train_dataset))
    print("Test length:",len(test_dataset))
    print("Total length:",len(train_dataset+test_dataset))
    #####
    dt=torch.utils.data.ConcatDataset([train_dataset,test_dataset])
    # Calculate sizes for 70-30 split
    total_size = len(dt)
    train_size = int(0.7 * total_size)
    test_size = total_size - train_size  # This accounts for rounding

    # Split the dataset
    train_dataset, test_dataset = torch.utils.data.random_split(
        dt, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )
    #####

    print("Train length:",len(train_dataset))
    print("Test length:",len(test_dataset))
    print("Total length:",len(train_dataset+test_dataset))

    # Create dataloaders
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=nw)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
    #                                        shuffle=False, num_workers=nw)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=nw)

    # Class information
    # class_names = train_dataset.classes
    # with open('class_indices.json', 'w') as f:
    #     json.dump(train_dataset.class_to_idx, f, indent=4)

    class_names = ['NORMAL','PNEUMONIA']

    # Initialize model
    net = medmamba(num_classes=len(class_names))#,drop_rate=0.1) #,patch_size=32) #,dims=[24,48,96,192], dims_decoder=[192,96,48,24],depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1])
    net.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate,weight_decay=1e-3,betas=(0.9, 0.999))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',       # Monitor validation accuracy
    factor=0.5,      # Reduce LR by half when plateaued
    patience=2,      # Wait 2 epochs without improvement
    verbose=True     # Print LR updates
)


    # Model paths
    model_name = "medmamba"
    best_model_path = f'./{model_name}_best.pth'
    last_model_path = f'./{model_name}_last.pth'
    
    # Load previous model if requested
    start_epoch = 0
    best_acc = 0.0
    train_loss_history, train_acc_history = [], []
    val_loss_history, val_acc_history = [], []
    
    if use_prev_trained and os.path.exists(last_model_path):
        net.load_state_dict(torch.load(last_model_path))
        if os.path.exists('./training_state.pth'):
            state = torch.load('./training_state.pth')
            start_epoch = state['epoch'] + 1
            best_acc = state['best_acc']
            train_loss_history = state['train_loss_history']
            train_acc_history = state['train_accuracy_history']
            val_loss_history = state['val_loss_history']
            val_acc_history = state['val_accuracy_history']
            print(f"Resuming training from epoch {start_epoch}")

    # Training loop
    if train:
      for epoch in range(start_epoch, epochs):
          net.train()
          running_loss = 0.0
          running_correct = 0
          
          train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                          bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
          
          for images, labels in train_bar:
              images, labels = images.to(device), labels.to(device)
              
              optimizer.zero_grad()
              outputs = net(images)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()
              
              _, preds = torch.max(outputs, 1)
              running_loss += loss.item() * images.size(0)
              running_correct += torch.sum(preds == labels.data)
              
              # Safe metric calculation
              current_batches = train_bar.n + 1  # Add 1 to prevent division by zero
              avg_loss = safe_divide(running_loss, current_batches * batch_size)
              avg_acc = safe_divide(running_correct.double(), current_batches * batch_size)
              
              train_bar.set_postfix({
                  'loss': f'{avg_loss:.4f}',
                  'acc': f'{avg_acc:.4f}'
              })
          
          # Calculate epoch metrics
          epoch_loss = safe_divide(running_loss, len(train_dataset))
          epoch_acc = safe_divide(running_correct.double(), len(train_dataset))
          train_loss_history.append(epoch_loss)
          train_acc_history.append(epoch_acc.item())
          
          # Validation
          net.eval()
          val_running_loss = 0.0
          val_running_correct = 0
          val_preds = []
          val_true = []
          
          with torch.no_grad():
              val_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]',
                            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
              
              for images, labels in val_bar:
                  images, labels = images.to(device), labels.to(device)
                  outputs = net(images)
                  loss = criterion(outputs, labels)
                  
                  _, preds = torch.max(outputs, 1)
                  val_running_loss += loss.item() * images.size(0)
                  val_running_correct += torch.sum(preds == labels.data)
                  val_preds.extend(preds.cpu().numpy())
                  val_true.extend(labels.cpu().numpy())
                  
                  # Safe metric calculation
                  current_batches = val_bar.n + 1
                  avg_loss = safe_divide(val_running_loss, current_batches * batch_size)
                  avg_acc = safe_divide(val_running_correct.double(), current_batches * batch_size)
                  
                  val_bar.set_postfix({
                      'loss': f'{avg_loss:.4f}',
                      'acc': f'{avg_acc:.4f}'
                  })
          
          val_epoch_loss = safe_divide(val_running_loss, len(test_dataset))
          val_epoch_acc = safe_divide(val_running_correct.double(), len(test_dataset))
          val_loss_history.append(val_epoch_loss)
          val_acc_history.append(val_epoch_acc.item())

          scheduler.step(val_epoch_acc)  
          
          print(f'\nEpoch {epoch+1}/{epochs}: '
                f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | '
                f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
          
          # Save model if improved
          if val_epoch_acc > best_acc:
              best_acc = val_epoch_acc
              torch.save(net.state_dict(), best_model_path)
              print(f"New best model saved with accuracy {best_acc:.4f}")
          
          # Save after every epoch
          if save_every_epoch:
              torch.save(net.state_dict(), last_model_path)
              torch.save({
                  'epoch': epoch,
                  'best_acc': best_acc,
                  'train_loss_history': train_loss_history,
                  'train_accuracy_history': train_acc_history,
                  'val_loss_history': val_loss_history,
                  'val_accuracy_history': val_acc_history
              }, './training_state.pth')
      
      # Plot training curves
      plot_metrics(epochs, train_loss_history, val_loss_history, 'Loss')
      plot_metrics(epochs, train_acc_history, val_acc_history, 'Accuracy')
      
    # Test evaluation
    print("\nEvaluating on test set...")
    net.load_state_dict(torch.load(best_model_path))
    net.eval()
    
    test_preds = []
    test_true = []
    test_running_correct = 0
    test_running_loss = 0.0
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing', 
                       bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            test_running_correct += torch.sum(preds == labels.data)
            test_running_loss += loss.item() * images.size(0)
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
            
            # Safe metric calculation
            current_batches = test_bar.n + 1
            avg_loss = safe_divide(test_running_loss, current_batches * batch_size)
            avg_acc = safe_divide(test_running_correct.double(), current_batches * batch_size)
            
            test_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.4f}'
            })
    
    test_loss = safe_divide(test_running_loss, len(test_dataset))
    test_acc = safe_divide(test_running_correct.double(), len(test_dataset))
    
    print(f'\nTest Results: Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}')
    print('\nClassification Report:')
    print(classification_report(test_true, test_preds, target_names=class_names))
    
    # Plot confusion matrix for test set
    plot_confusion_matrix(test_true, test_preds, class_names, 'Test Set Confusion Matrix')

if __name__ == '__main__':
    main()