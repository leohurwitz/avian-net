from tqdm.auto import tqdm
import torch
import torch.nn as nn
from src.model import AvianNetModelV1, AvianNetModelV2
from src.dataset import get_dataloaders
from timeit import default_timer as timer
from pathlib import Path
from src.config_loader import load_config


CONFIG = load_config()

def train_step(model, dataloader, loss_fn, optimizer, device):
    # Batch Loop for a single training epoch
    model.train()
    train_loss = 0
    # X is the spectrograms, y is the label
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        
        # Forward Pass
        y_pred = model(X)
        # Calculate Loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() # accumulate train loss across batches

        # Clear optimizer's memory
        optimizer.zero_grad()

        # Backward pass to find weights that cause error
        loss.backward()
    
        # Call .step() to adjust those weights
        optimizer.step()


    # Divide total train loss by length of train dataloader
    train_loss /= len(dataloader)
    return train_loss

def test_step(model, dataloader, loss_fn, scheduler, device):

    test_loss, test_acc = 0, 0
    test_correct = 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in dataloader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            # Forward Pass
            test_pred = model(X_test)

            # Calculate accumulative loss
            test_loss += loss_fn(test_pred, y_test).item()

            # Calculate accuracy
            test_pred_labels = test_pred.argmax(dim=1)
            test_correct += (test_pred_labels == y_test).sum().item()
        

        # Calculate test loss avg per batch
        test_loss /= len(dataloader)
        scheduler.step(test_loss)

        # Calculate test accuracy per batch
        test_acc = (test_correct / len(dataloader.dataset)) * 100
        return test_loss, test_acc

def calculate_class_weights(train_dataloader):
# Implementing Class Weights
    class_counts = torch.zeros(5)
    for _, y in train_dataloader:
        for label in y:
            class_counts[label] += 1

    total_samples = class_counts.sum()
    num_classes = len(class_counts)
    class_weights = (total_samples / (num_classes * class_counts))

    return class_weights

def initialize_model(device):
    if CONFIG['training']['model'] == 'v1':
        # Custom CNN Architecture
        model = AvianNetModelV1(
            input_shape=3, 
            hidden_units=32, 
            output_shape=5).to(device)
        return model
    
    if CONFIG['training']['model'] == 'v2':
        # Transfer Learning using ResNet18 (Feature Extraction --> Frozen Weights)
        freeze_weights = CONFIG['training']['freeze_weights']
        model = AvianNetModelV2(num_classes=5, freeze_weights=True).to(device)
        return model

    if CONFIG['training']['model'] == 'v3':
        # Transfer Learning using ResNet18 (Fine-Tuning --> Unfrozen Weights)
        freeze_weights = CONFIG['training']['freeze_weights']
        model = AvianNetModelV2(num_classes=5, freeze_weights=False).to(device)
        return model

def fit(model, train_dataloader, test_dataloader, loss_fn, optimizer, scheduler, device, epochs, patience):

    best_loss = float('inf')
    failure_times = 0

    # Runs Epoch Loop
    for epoch in tqdm(range(epochs)):
            # Runs Train/Test Loop
            print(f"\nEpoch: {epoch}\n--------")
            print(f"Current LR: {optimizer.param_groups[0]['lr']}")
            train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
            test_loss, test_acc = test_step(model, test_dataloader, loss_fn, scheduler, device)
            print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

            # Checkpoint Model Saving
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), Path.cwd() / 'models' / f'avian_net_{CONFIG['training']['model']}_loss_scheduler.pth')
                failure_times = 0
                print(f"Model Improved: Weights Saved\n\n\n")
            else: # Early Stopping to prevent overfitting
                failure_times += 1
                print(f"No Improvement. Patience: {failure_times}/{patience}\n\n\n")

                if failure_times >= patience:
                    print(f"Early Stopping. Validation hasn't improved for {patience} epochs")
                    break
    
def main():
    torch.manual_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data Setup
    train_dataloader, test_dataloader = get_dataloaders()

    class_weights = calculate_class_weights(train_dataloader)

    # Model Setup
    model = initialize_model(device=device)

    # Loss & Optimizer Setup
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad], 
        lr=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    # Running Model
    start_time = timer()

    fit(model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=CONFIG['training']['epochs'],
        patience=CONFIG['training']['patience']
    )

    # Tracks Training Time
    if device.type == 'mps': 
        torch.mps.synchronize()
        end_time = timer()
        total_train_time = end_time - start_time
        print(f"Train time on {device}: {total_train_time:.2f} seconds")
    
    
if __name__ == '__main__':
    main()