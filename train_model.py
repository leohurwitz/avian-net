from tqdm.auto import tqdm
import torch
import torch.nn as nn
from src.model import AvianNetModelV1
from src.dataset import get_dataloaders
from timeit import default_timer as timer


# Model Training
if __name__ == "__main__":
    start_time = timer()
    # Loop over specified number of epochs
    torch.manual_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    epochs = 3
    train_dataloader, test_dataloader = get_dataloaders()

    model_v1 = AvianNetModelV1(input_shape=3, 
                                hidden_units=10,
                                output_shape=5).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_v1.parameters(), lr=0.01, momentum=0.9, nesterov=True)


    for epoch in tqdm(range(epochs)):
        print(f"\nEpoch: {epoch}\n--------")

        ### Training
        model_v1.train()
        train_loss = 0
        # X is the spectrograms, y is the label
        for batch, (X, y) in enumerate(train_dataloader):
            X = X.to(device)
            y = y.to(device)
            
            # Forward Pass
            y_pred = model_v1(X)
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
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

        # Divide total train loss by length of train dataloader
        train_loss /= len(train_dataloader)

        ### Testing
        test_loss, test_acc = 0, 0
        test_correct = 0
        model_v1.eval()
        with torch.inference_mode():
            for X_test, y_test in test_dataloader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                # Forward Pass
                test_pred = model_v1(X_test)

                # Calculate accumulative loss
                test_loss += loss_fn(test_pred, y_test).item()

                # Calculate accuracy
                test_pred_labels = test_pred.argmax(dim=1)
                test_correct += (test_pred_labels == y_test).sum().item()
            

            # Calculate test loss avg per batch
            test_loss /= len(test_dataloader)

            # Calculate test accuracy per batch
            test_acc = (test_correct / len(test_dataloader.dataset)) * 100
        
        # Print loss
        print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
    
    if device.type == 'mps':
        torch.mps.synchronize()
    end_time = timer()
    total_train_time = end_time - start_time
    print(f"Train time on {device}: {total_train_time:.2f} seconds")


