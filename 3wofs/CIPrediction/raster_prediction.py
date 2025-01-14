# -*- coding: utf-8 -*-

import os
import torch
from raster_dataset import LoadData
from torch.utils.data import DataLoader
import warnings
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from gcnnet import GCN
from chebnet import ChebNet
from gat import GATNet
import torch.nn as nn
import torch.optim as optim
from utils import Evaluation
import argparse
print(torch.__version__)
print(torch.cuda.is_available())


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Configure the GPU
    # Argument parsing for --model
    parser = argparse.ArgumentParser(description="Model selection for Graph Neural Networks")
    parser.add_argument('--model', type=str, choices=['GCN', 'GATNet', 'ChebNet'], default='GCN', help="Choose the model: GCN, GATNet, or ChebNet")
    args = parser.parse_args()
    # Step 1: Load the data
    train_data = LoadData(num_nodes=1024, train_mode="train", graph_type="distance")
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True,
                              num_workers=32)  # num_workers is the number of threads for loading data (batches)

    test_data = LoadData(num_nodes=1024, train_mode="test", graph_type="distance")
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=32)
    valid_data = LoadData(num_nodes=1024, train_mode="val", graph_type="distance")
    valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False, num_workers=8)

    # Step 2: Define the model (we assume the model is already defined, just loading it here)
    in_c = 4
    hid_c = 16
    out_c = 16
    fc_n = 3
    Epoch = 100  # Number of training epochs
    # Choose the network type
    if args.model == "GCN":
        # GCN Model
        my_net = GCN(in_c=in_c, hid_c=hid_c, out_c=out_c)  # Load GCN model
        result_file = "GCN.h5"
        savefig = "GCN"
        model_path = "gcn"
    if args.model == "ChebNet":
        in_c = 6
        hid_c = 16
        out_c = 16
        K = 2
        my_net = ChebNet(in_c=in_c, hid_c=hid_c, out_c=out_c, K=K)  # Load ChebNet model
        result_file = "CN.h5"
        savefig = "CN"
        model_path = "cn"
    elif args.model == "GATNet":
        in_c = 4
        hid_c = 32
        out_c = 32
        n_heads = 4
        my_net = GATNet(in_c=in_c, hid_c=hid_c, out_c=out_c, n_heads=n_heads)  # Load GAT model

        result_file = "GAT.h5"
        savefig = "GAT"
        model_path = "gat"

    print(my_net)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # Define the device

    my_net = my_net.to(device)

    # Step 3: Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = optim.Adam(params=my_net.parameters(), lr=0.001)  # Use Adam optimizer with default learning rate

    # Step 4: Training + Testing
    # Train model
    x = []

    train_losses = []
    train_r2 = []
    # Set L1 regularization coefficient
    l1_lambda = 0.01
    my_net.train()  # Set model to training mode
    for epoch in range(Epoch):
        epoch_loss = 0.0
        count = 0
        y_predict = []
        y_true = []
        y_valid_predict = []
        y_valid_true = []
        start_time = time.time()
        x.append(epoch)
        for data in train_loader:
            optimizer.zero_grad()
            count += 1
            predict_value = my_net(data, device).to(torch.device("cpu"))
            print("predict_value", predict_value)
            l1_loss = l1_lambda * sum(torch.abs(param).sum() for param in my_net.parameters())

            loss = criterion(predict_value, data["vertices_feature_y"])

            # Add L1 regularization term to the loss
            total_loss = loss + l1_loss

            epoch_loss += loss.item()

            # Backpropagation
            total_loss.backward()
            labels_array = data["vertices_feature_y"]
            outputs_array = predict_value.detach().numpy().flatten()
            for ele in labels_array: y_true.append(ele)
            for ele in outputs_array: y_predict.append(ele)
            optimizer.step()

        # Calculate validation loss and R2 score
        my_net.eval()  # Set model to evaluation mode
        valid_loss = 0.0
        with torch.no_grad():
            for valid_data in valid_loader:
                predict_value_valid = my_net(valid_data, device).to(torch.device("cpu"))

                loss_valid = criterion(predict_value_valid, valid_data["vertices_feature_y"])
                valid_loss += loss_valid.item()

                y_valid_true.extend(valid_data["vertices_feature_y"].numpy().flatten())
                y_valid_predict.extend(predict_value_valid.numpy().flatten())
        my_net.train()  # Switch back to training mode

        end_time = time.time()
        train_losses.append(epoch_loss / len(train_loader))
        r2 = Evaluation.r2_(y_true, y_predict)
        train_r2.append(r2)
        print("Epoch: {:04d}, Loss: {:.4f}, R2: {:.4f}, Time: {:.2f} mins".format(
            epoch, epoch_loss / len(train_loader), r2, (end_time - start_time) / 60))
        plt.figure(figsize=(3, 6), dpi=300)
        # Create a 2x1 plot and plot the first graph
        plt.subplot(2, 1, 1)
        try:
            train_loss_lines.remove(train_loss_lines[0])  # Remove previous loss line
        except Exception:
            pass
        train_loss_lines = plt.plot(x, train_losses, label='Train Loss', color='blue')  # Line width (lw)
        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.xlim(0, Epoch)
        plt.ylim(0, 2)

        # Plot the second graph (R2)
        plt.subplot(2, 1, 2)
        try:
            train_r2_lines.remove(train_r2_lines[0])  # Remove previous R2 line
        except Exception:
            pass
        train_r2_lines = plt.plot(x, train_r2, label='R2', color='blue')  # Line width (lw)
        plt.title("r2")
        plt.xlabel("epoch")
        plt.ylabel("r2")
        plt.legend()
        plt.xlim(0, Epoch)
        plt.ylim(0, 1)
        plt.text(60, 0.4, 'epoch:' + str(Epoch))
        plt.text(60, 0.3, 'in_c:' + str(in_c))
        plt.text(60, 0.2, 'hid_c:' + str(hid_c))
        plt.text(60, 0.1, 'out_c:' + str(out_c))
        if net == "GATNet":
            plt.text(60, 0.05, 'n_heads:' + str(n_heads))
        plt.show()
        plt.pause(0.1)  # Pause for 0.1s to allow the plot to refresh

    my_net.eval()  # Switch to test mode
    with torch.no_grad():  # Disable gradient computation
        MAE, MAPE, RMSE = [], [], []  # Define lists for the three metrics
        Target = np.zeros([1])  # [N, T, D], T=1 # Fill the target data with zeros
        Predict = np.zeros_like(Target)  # [N, T, D], T=1 # Predictions initialized to zeros

        total_loss = 0.0
        for data in test_loader:  # Iterate through test data batch by batch
            # The following prediction is in the normalized form, but we need to reverse the normalization for evaluation
            predict_value = my_net(data, device).to(torch.device("cpu"))  # [B, N, 1, D]
            loss = criterion(predict_value, data["vertices_feature_y"])  # Calculate MSE loss

            total_loss += loss.item()  # Accumulate total loss
            # Flatten the predictions and target values
            target_value = data["vertices_feature_y"]  # [64]

            performance, data_to_save = compute_performance(predict_value, target_value,
                                                            test_loader)  # Compute model performance

            # Concatenate predictions and targets for the entire batch
            Predict = np.concatenate([Predict, data_to_save[0]])
            Target = np.concatenate([Target, data_to_save[1]])

            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])
            R2.append(performance[3])

        print("Test Loss: {:02.4f}".format(total_loss / len(test_data)))

    # Calculate the mean for all three metrics
    print("Performance:  MAE {:2.2f}  MAPE  {:2.2f}%  RMSE  {:2.2f}".format(np.mean(MAE), np.mean(MAPE * 100),
                                                                            np.mean(RMSE)))
    # Save results
    file_obj = h5py.File("path/to/" + result_file, "w")
    file_obj["predict"] = Predict  # Store predicted values
    file_obj["target"] = Target


def compute_performance(prediction, target, data):  # Compute model performance
    # The following try-except block essentially does this: when training and testing the model,
    # the data is passed through DataLoader, so it can be assigned directly.
    # However, if the trained model is saved and then tested, the data has not gone through DataLoader,
    # so it needs to be converted from DataLoader to Dataset type.
    try:
        dataset = data.dataset  # If the data is of type DataLoader, convert it to Dataset type via its .dataset attribute
    except:
        dataset = data  # If the data is already a Dataset, directly assign it

    # The following line calls a method to compute three evaluation metrics.
    # This method is encapsulated in a separate file and is used here.
    mae, mape, rmse, r2 = Evaluation.total(target.numpy(), prediction.numpy())  # Convert tensors to NumPy arrays to compute metrics

    performance = [mae, mape, rmse, r2]  # Store the computed performance metrics
    recovered_data = [prediction, target]  # Store the recovered data (for visualization)

    return performance, recovered_data  # Return performance results and recovered data for visualization

if __name__ =='__main__':
    main()


