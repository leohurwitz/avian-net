from tqdm.auto import tqdm
import torch
import torch.nn as nn
from src.model import AvianNetModelV1
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

        if batch % 20 == 0:
            print(f"Looked at {batch * len(X)}/{len(dataloader.dataset)} samples")

    # Divide total train loss by length of train dataloader
    train_loss /= len(dataloader)
    return train_loss

def test_step(model, dataloader, loss_fn, device):

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

        # Calculate test accuracy per batch
        test_acc = (test_correct / len(dataloader.dataset)) * 100
        return test_loss, test_acc


if __name__ == '__main__':
    start_time = timer()
    torch.manual_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    train_dataloader, test_dataloader = get_dataloaders()

    # Implementing Class Weights
    class_counts = torch.zeros(5)
    for _, y in train_dataloader:
        for label in y:
            class_counts[label] += 1

    total_samples = class_counts.sum()
    num_classes = len(class_counts)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights.to(device)

    epochs = CONFIG['training']['epochs']
    lr = CONFIG['training']['learning_rate']
    weight_decay = CONFIG['training']['weight_decay']
        

    model_v1 = AvianNetModelV1(input_shape=3, 
                                hidden_units=32,
                                output_shape=5).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(params=model_v1.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float('inf')
    patience = CONFIG['training']['patience']   
    failure_times = 0
    
    for epoch in tqdm(range(epochs)):
            print(f"\nEpoch: {epoch}\n--------")
            train_loss = train_step(model_v1, train_dataloader, loss_fn, optimizer, device)
            test_loss, test_acc = test_step(model_v1, test_dataloader, loss_fn, device)
            print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model_v1.state_dict(), Path.cwd() / 'models' / 'avian_net_V1.pth')
                failure_times = 0
                print(f"Model Improved: Weights Saved")
            else:
                failure_times += 1
                print(f"No Improvement. Patience: {failure_times}/{patience}")
                if failure_times >= patience:
                    print(f"Early Stopping. Validation hasn't improved for {patience} epochs")
                    break

    if device.type == 'mps':
        torch.mps.synchronize()
        end_time = timer()
        total_train_time = end_time - start_time
        print(f"Train time on {device}: {total_train_time:.2f} seconds")



