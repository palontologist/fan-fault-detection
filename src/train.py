import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json

from data_processing import create_dataloaders
from model import create_model, CNNAutoencoder


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: dict
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (data, labels) in enumerate(pbar):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        reconstructed, latent = model(data)
        loss = criterion(reconstructed, data)
        
        loss.backward()
        
        if config['training']['hyperparameters']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['hyperparameters']['gradient_clip']
            )
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return {'loss': total_loss / num_batches}


def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_errors = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for data, labels in pbar:
            data = data.to(device)
            labels = labels.to(device)
            
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            
            errors = model.get_reconstruction_error(data)
            all_errors.extend(errors.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
    
    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)
    
    threshold = np.percentile(
        all_errors[all_labels == 0],
        config['training']['loss']['anomaly_threshold_percentile']
    )
    
    predictions = (all_errors > threshold).astype(int)
    
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    
    auc = roc_auc_score(all_labels, all_errors)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average='binary'
    )
    
    return {
        'loss': total_loss / num_batches,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }


def train(config: dict):
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    model = create_model(config).to(device)
    
    criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['hyperparameters']['learning_rate'],
        weight_decay=config['training']['hyperparameters']['weight_decay']
    )
    
    scheduler_type = config['training']['hyperparameters']['scheduler']
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['hyperparameters']['epochs']
        )
    elif scheduler_type == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
    else:
        scheduler = None
    
    writer = SummaryWriter(log_dir=config['training'].get('log_dir', 'runs/fan_fault_detection'))
    
    best_auc = 0.0
    patience_counter = 0
    early_stopping_patience = config['training']['hyperparameters']['early_stopping_patience']
    
    checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, config['training']['hyperparameters']['epochs'] + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch, config)
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('AUC/val', val_metrics['auc'], epoch)
        writer.add_scalar('Precision/val', val_metrics['precision'], epoch)
        writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        writer.add_scalar('Threshold/val', val_metrics['threshold'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val AUC: {val_metrics['auc']:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}")
        
        if scheduler is not None:
            if scheduler_type == "reduce_on_plateau":
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'threshold': val_metrics['threshold'],
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f"  -> New best model saved! AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
    
    writer.close()
    
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(checkpoint_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    threshold = checkpoint['threshold']
    
    test_metrics = evaluate_test(model, test_loader, criterion, device, threshold)
    
    results = {
        'best_val_auc': best_auc,
        'test_auc': test_metrics['auc'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'threshold': threshold
    }
    
    with open(checkpoint_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal Results:")
    print(f"  Best Val AUC: {best_auc:.4f}")
    print(f"  Test AUC: {test_metrics['auc']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall: {test_metrics['recall']:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    
    return model, threshold


def evaluate_test(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float
) -> Dict[str, float]:
    model.eval()
    all_errors = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Testing"):
            data = data.to(device)
            labels = labels.to(device)
            
            errors = model.get_reconstruction_error(data)
            all_errors.extend(errors.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)
    
    predictions = (all_errors > threshold).astype(int)
    
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    
    auc = roc_auc_score(all_labels, all_errors)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average='binary'
    )
    
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config['training']['log_dir'] = 'runs/fan_fault_detection'
    config['training']['checkpoint_dir'] = 'checkpoints'
    config['data']['data_dir'] = 'data/processed'
    
    train(config)