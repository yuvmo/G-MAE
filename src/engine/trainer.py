import torch
import torch.optim as optim
from tqdm import tqdm
from utils.logger import log_to_wandb
import os



class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = torch.amp.GradScaler(enabled=config.fp16)
        self.best_val_loss = float('inf')
        self.autocast = torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config.fp16)
        
    def _create_optimizer(self):
        return optim.AdamW(
            self.model.parameters(), 
            lr=float(self.config.lr), 
            weight_decay=self.config.weight_decay
        )
        
    def _create_scheduler(self):
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.epochs
        )
        
    def train_epoch(self, epoch):
        raise NotImplementedError
        
    def validate_epoch(self):
        raise NotImplementedError
        
    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.save_dir, "best_model.pth"))
        else:
            torch.save(checkpoint, os.path.join(self.config.save_dir, f"checkpoint_{epoch}.pth"))
            
    def train(self):
        raise NotImplementedError


class PreTrainer(BaseTrainer):
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}/{self.config.epochs}")
        
        for batch_idx, images in enumerate(progress_bar):
            images = images.to(self.device)
            
            with self.autocast:
                loss = self.model(images)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            if self.config.log_to_wandb and batch_idx % 10 == 0:
                log_to_wandb({
                    "train/loss": loss.item(),
                    "train/lr": self.optimizer.param_groups[0]['lr']
                }, step=epoch * len(self.train_loader) + batch_idx)
        
        self.scheduler.step()
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        for images in progress_bar:
            images = images.to(self.device)
            
            with self.autocast:
                loss = self.model(images)
            
            total_loss += loss.item()
            progress_bar.set_postfix(val_loss=loss.item())
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        print("Starting GMAE pre-training...")
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch()
            
            print(f"Epoch {epoch+1}/{self.config.epochs}, "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if self.config.log_to_wandb:
                log_to_wandb({
                    "epoch": epoch+1,
                    "train/avg_loss": train_loss,
                    "val/avg_loss": val_loss,
                    "train/lr": self.optimizer.param_groups[0]['lr']
                })
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(f"New best model saved with val loss: {val_loss:.4f}")
            
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(epoch, val_loss)
                print(f"Checkpoint saved at epoch {epoch+1}")


class FinetuneTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, config, device):
        super().__init__(model, train_loader, val_loader, config, device)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        processed_batches = 0
        
        if len(self.train_loader) == 0:
            print("Warning: Train loader is empty!")
            return 0.0, 0.0
            
        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}/{self.config.epochs}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            try:
                images = images.to(self.device)
                targets = targets.to(self.device).long()
                
                if epoch < self.config.freeze_encoder_epochs:
                    with torch.no_grad():
                        encoder_output = self.model.encoder(images)
                        features = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output
                    logits = self.model.classifier(features)
                else:
                    with self.autocast:
                        model_output = self.model(images)
                        logits = model_output[0] if isinstance(model_output, tuple) else model_output
                
                if logits.dim() > 2:
                    logits = logits.view(logits.size(0), -1)
                
                if logits.size(0) != targets.size(0):
                    print(f"Error: Batch size mismatch! Logits: {logits.shape}, Targets: {targets.shape}")
                    continue
                
                loss = self.criterion(logits, targets)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                batch_size = targets.size(0)
                total += batch_size
                correct += (predicted == targets).sum().item()
                processed_batches += 1
                
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': f'{100 * correct / total:.2f}%' if total > 0 else 'N/A'
                })
                
                if self.config.log_to_wandb and batch_idx % 10 == 0 and total > 0:
                    log_to_wandb({
                        "train/loss": loss.item(),
                        "train/acc": 100 * correct / total,
                        "train/lr": self.optimizer.param_groups[0]['lr']
                    }, step=epoch * len(self.train_loader) + batch_idx)
            
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                print(f"Images shape: {images.shape if 'images' in locals() else 'N/A'}")
                print(f"Targets shape: {targets.shape if 'targets' in locals() else 'N/A'}")
                if 'logits' in locals():
                    print(f"Logits shape: {logits.shape}")
                continue
        
        if processed_batches == 0:
            print("Warning: No batches processed in this epoch!")
            return 0.0, 0.0
        
        self.scheduler.step()
        avg_loss = total_loss / processed_batches
        avg_acc = 100 * correct / total
        return avg_loss, avg_acc

    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        processed_batches = 0
        
        if len(self.val_loader) == 0:
            print("Warning: Validation loader is empty!")
            return 0.0, 0.0
            
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        for images, targets in progress_bar:
            try:
                images = images.to(self.device)
                targets = targets.to(self.device).long()
                
                with self.autocast:
                    model_output = self.model(images)
                    logits = model_output[0] if isinstance(model_output, tuple) else model_output
                    
                    if logits.dim() > 2:
                        logits = logits.view(logits.size(0), -1)
                        
                    if logits.size(0) != targets.size(0):
                        print(f"Error: Batch size mismatch! Logits: {logits.shape}, Targets: {targets.shape}")
                        continue
                    
                    loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                batch_size = targets.size(0)
                total += batch_size
                correct += (predicted == targets).sum().item()
                processed_batches += 1
                
                progress_bar.set_postfix({
                    'val_loss': loss.item(),
                    'val_acc': f'{100 * correct / total:.2f}%' if total > 0 else 'N/A'
                })
            
            except Exception as e:
                print(f"Error processing validation batch: {str(e)}")
                continue
        
        if processed_batches == 0:
            print("Warning: No batches processed in validation!")
            return 0.0, 0.0
        
        avg_loss = total_loss / processed_batches
        avg_acc = 100 * correct / total
        return avg_loss, avg_acc

    def train(self):
        print("Starting GMAE fine-tuning...")
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch()
            
            print(f"Epoch {epoch+1}/{self.config.epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if self.config.log_to_wandb:
                log_to_wandb({
                    "epoch": epoch+1,
                    "train/avg_loss": train_loss,
                    "train/avg_acc": train_acc,
                    "val/avg_loss": val_loss,
                    "val/avg_acc": val_acc,
                    "train/lr": self.optimizer.param_groups[0]['lr']
                })
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(f"New best model saved with val loss: {val_loss:.4f}, val acc: {val_acc:.2f}%")
            
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(epoch, val_loss)
                print(f"Checkpoint saved at epoch {epoch+1}")