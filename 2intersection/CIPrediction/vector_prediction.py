import os
import torch
from vector_dataset import LoadData
from torch.utils.data import DataLoader
import warnings
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
warnings.filterwarnings('ignore')
from gcnnet import GCN
from chebnet import ChebNet
from gat import GATNet
import torch.nn as nn
import torch.optim as optim
from utils import Evaluation
print(torch.__version__)
print(torch.cuda.is_available())
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Configuring the GPU
    # Argument parsing for --model
    parser = argparse.ArgumentParser(description="Model selection for Graph Neural Networks")
    parser.add_argument('--model', type=str, choices=['GCN', 'GATNet', 'ChebNet'], default='GCN', help="Choose the model: GCN, GATNet, or ChebNet")
    args = parser.parse_args()
    # Step 1: Load the data
    train_data = LoadData(num_nodes=1024, train_mode="train", graph_type="distance")
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=32)

    test_data = LoadData(num_nodes=1024, train_mode="test", graph_type="distance")
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=32)

    # Step 2: Define the model based on the argument
    if args.model == "GCN":
        my_net = GCN(in_c=3, hid_c=8, out_c=16)  # GCN model
        result_file = "GCN.h5"
        savefig = "GCN"
        model_path = "gcn"
    elif args.model == "ChebNet":
        my_net = ChebNet(in_c=3, hid_c=8, out_c=16, K=2)  # ChebNet model
        result_file = "CN.h5"
        savefig = "CN"
        model_path = "cn"
    elif args.model == "GATNet":
        my_net = GATNet(in_c=3, hid_c=4, out_c=4, n_heads=4)  # GATNet model
        result_file = "GAT.h5"
        savefig = "GAT"
        model_path = "gat"

    print(my_net)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Define the device (GPU or CPU)

    my_net = my_net.to(device)

    # Step 3: Define the loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(params=my_net.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

    # Step 4: Train and Test the model
    # Train model
    x = []
    Epoch = 100  # Number of training epochs
    train_losses = []
    train_r2 = []
    my_net.train()  # Set the model to training mode
    for epoch in range(Epoch):
        epoch_loss = 0.0
        count = 0
        y_predict = []
        y_true = []
        start_time = time.time()
        x.append(epoch)
        for data in train_loader:
            my_net.zero_grad()
            count += 1
            predict_value = my_net(data, device).to(torch.device("cpu"))
            # print("predict_value",predict_value)
            loss = criterion(predict_value, data["vertices_feature_y"])

            epoch_loss += loss.item()

            loss.backward()
            labels_array = data["vertices_feature_y"]
            outputs_array = predict_value.detach().numpy().flatten()
            for ele in labels_array: y_true.append(ele)
            for ele in outputs_array: y_predict.append(ele)
            optimizer.step()
        end_time = time.time()
        train_losses.append(epoch_loss / len(train_data))
        r2 = Evaluation.r2_(y_true, y_predict)
        train_r2.append(r2)
        print("Epoch: {:04d}, Loss: {:02.4f}, R2: {:02.4f}, Time: {:02.2f} mins".format(
            epoch, epoch_loss / len(train_data), r2, (end_time - start_time) / 60))

        # Plot loss and R2 during training
        plt.figure(figsize=(3, 6), dpi=300)
        plt.subplot(2, 1, 1)  # Create a plot for loss
        try:
            train_loss_lines.remove(train_loss_lines[0])  # Remove the previous curve
        except Exception:
            pass
        train_loss_lines = plt.plot(x, train_losses, 'r', lw=1)  # Plot loss curve
        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend("loss")
        plt.xlim(2, 200)
        plt.ylim(0, 0.03)

        plt.subplot(2, 1, 2)  # Create a plot for R2
        try:
            train_r2_lines.remove(train_r2_lines[0])  # Remove the previous curve
        except Exception:
            pass
        train_r2_lines = plt.plot(x, train_r2, 'r', lw=1)  # Plot R2 curve
        plt.title("r2")
        plt.xlabel("epoch")
        plt.ylabel("r2")
        plt.legend("train_r2")
        plt.xlim(2, 200)
        plt.ylim(0, 1)
        plt.show()
        plt.pause(0.1)  # Pause the plot for 0.1s

    my_net.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for testing
        MAE, MAPE, RMSE = [], [], []  # Define lists for evaluation metrics
        Target = np.zeros([1])  # [N, T, D], T=1, Initialize with zeros
        Predict = np.zeros_like(Target)  # [N, T, D], T=1, Initialize with zeros

        total_loss = 0.0
        for data in test_loader:  # Fetch one batch of test data at a time
            predict_value = my_net(data, device).to(torch.device("cpu"))  # Predict values
            loss = criterion(predict_value, data["vertices_feature_y"])  # Calculate MSE loss

            total_loss += loss.item()  # Accumulate loss for the entire batch
            target_value = data["vertices_feature_y"]  # Get the target values

            performance, data_to_save = compute_performance(predict_value, target_value, test_loader)  # Compute performance metrics

            # Concatenate the prediction and target data for visualization
            Predict = np.concatenate([Predict, data_to_save[0]])
            Target = np.concatenate([Target, data_to_save[1]])

            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])
            print("Test Loss: {:02.4f}".format(total_loss / len(test_data)))

    # Print average performance metrics
    print("Performance:  MAE {:2.2f}  MAPE  {:2.2f}%  RMSE  {:2.2f}".format(
        np.mean(MAE), np.mean(MAPE * 100), np.mean(RMSE)))

    file_obj = h5py.File(result_file, "w")  # Save the prediction and target values to a file for visualization

    file_obj["predict"] = Predict  # Save predicted values
    file_obj["target"] = Target  # Save target values

    # Save the trained model
    torch.save(my_net.state_dict(), "path/to/" + model_path + ".pth")



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