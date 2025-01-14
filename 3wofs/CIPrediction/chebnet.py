import torch
import torch.nn as nn
import torch.nn.init as init


class ChebConv(nn.Module):
    """

    :param in_c:
    :param out_c:
    :param K:
    :param bias:
    :param normalize:
    """
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):

        super(ChebConv, self).__init__()
        self.normalize = normalize  # Regularization parameter, True or False
        # [K+1, 1, in_c, out_c], the second 1 is for dimensional expansion, used for convenience in calculation. Whether it is there or not does not affect the parameter size. nn.Parameter is used to make the parameter a trainable model parameter.
        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))
        # The reason for using K+1 is that K starts from 0.
        init.xavier_normal_(self.weight)  # Initialize with a normal distribution

        if bias:  # Bias term, b in a linear function
            # The two 1s in the front are for simplicity, since the output dimension is 3.
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)  # Initialize with zeros
        else:
            self.register_parameter("bias", None)
        self.K=K+1

    def forward(self, inputs, graph):
        # [B, N, N], obtain the Laplacian matrix
        L = ChebConv.get_laplacian(graph, self.normalize)
        # [K, B, N, N], this is the multi-order Chebyshev polynomial. K is the order, N is the number of nodes.
        mul_l = self.cheb_polynomial(L).transpose(0, 1).contiguous()  # [B, K, N, N] -> [K, B, N, N]

        # [K, B, N, C], this is the result after multiplying with x
        result = torch.matmul(mul_l, inputs)
        # [K, B, N, D], multiply the result by W
        result = torch.matmul(result, self.weight)
        # [B, N, D], sum over the K dimension and add the bias
        result = torch.sum(result, dim=0) + self.bias
        return result

    # Compute Chebyshev polynomial, which is T_k(L) from the previous formula
    def cheb_polynomial(self, laplacian):
        B = laplacian.size(0)  # Number of batches
        N = laplacian.size(1)  # Number of nodes
        multi_order_laplacians = torch.zeros([B, self.K, N, N], device=laplacian.device, dtype=torch.float)

        for i in range(0, B):
            sub_laplacian = laplacian[i]  # Extract the sub-Laplacian for the batch
            N = sub_laplacian.size(0)  # Number of nodes

            # [K, N, N], initialize a zero matrix for the multi-order Laplacian polynomial
            multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)

            # 0th order Chebyshev polynomial is the identity matrix
            multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

            if self.K == 1:
                multi_order_laplacians[i] = multi_order_laplacian
                return multi_order_laplacians
            else:  # For order greater than or equal to 1
                multi_order_laplacian[1] = sub_laplacian  # 1st order Chebyshev polynomial is the Laplacian itself
                multi_order_laplacians[i] = multi_order_laplacian
                if self.K == 2:  # 1st order Chebyshev polynomial is just the Laplacian matrix L
                    return multi_order_laplacians
                else:
                    for k in range(2, self.K):
                        # Recurrence relation for Chebyshev polynomials: T_k(L) = 2 * L * T_{k-1}(L) - T_{k-2}(L)
                        multi_order_laplacian[k] = 2 * torch.mm(sub_laplacian, multi_order_laplacian[k - 1]) - \
                                                   multi_order_laplacian[k - 2]
                        multi_order_laplacians[i] = multi_order_laplacian
        return multi_order_laplacians

    @staticmethod
    # Compute the Laplacian matrix
    def get_laplacian(graph, normalize):
        b, n = graph.size(0), graph.size(1)  # b: batch size, n: number of nodes
        # print("Graph size:", graph.size())
        Ls = torch.zeros([b, n, n], device=graph.device, dtype=graph.dtype)  # Initialize Laplacian matrix

        for i in range(0, b):
            sub_graph = graph[i]  # Extract the subgraph for the batch
            if normalize:
                # The graph here is the adjacency matrix, D is the degree matrix
                D = torch.diag(torch.sum(sub_graph, dim=-1) ** (-1 / 2))  # Degree matrix D^{-1/2}
                D = torch.where(torch.isnan(D), torch.full_like(D, 0), D)  # Handle NaN values in D
                # L = I - D * A * D, this is the normalized Laplacian
                L = torch.eye(sub_graph.size(0), device=sub_graph.device, dtype=sub_graph.dtype) - torch.mm(
                    torch.mm(D, sub_graph), D)
                L = torch.where(torch.isnan(L), torch.full_like(L, 0), L)  # Handle NaN values in L
                L = torch.where(torch.isinf(L), torch.full_like(L, 0), L)  # Handle inf values in L
                Ls[i] = L
            else:
                D = torch.diag(torch.sum(sub_graph, dim=-1))  # Degree matrix D
                D = torch.where(torch.isnan(D), torch.full_like(D, 0), D)  # Handle NaN values in D
                L = D - sub_graph  # Unnormalized Laplacian: L = D - A
                L = torch.where(torch.isnan(L), torch.full_like(L, 0), L)  # Handle NaN values in L
                L = torch.where(torch.isinf(L), torch.full_like(L, 0), L)  # Handle inf values in L
                Ls[i] = L

        return Ls


# Define the Graph Neural Network class
class ChebNet(nn.Module):
    """
    ChebNet class for graph neural network using Chebyshev graph convolution.

    :param in_c: Input channels (number of features for each vertex)
    :param hid_c: Hidden channels (number of features for hidden layers)
    :param out_c: Output channels (number of features for output layer)
    :param K: Order of Chebyshev polynomial
    """

    def __init__(self, in_c, hid_c, out_c, K):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c, K=K)  # First graph convolutional layer
        self.conv2 = ChebConv(in_c=hid_c, out_c=out_c, K=K)  # Second graph convolutional layer
        self.fc = nn.Sequential(
            nn.Linear(512 * out_c, 512),  # Fully connected layer 1
            nn.Linear(512, 128),  # Fully connected layer 2
            nn.Linear(128, 1)  # Fully connected layer 3 (output layer)
        )  # Additional fully connected layer
        self.act = nn.ReLU()  # Activation function (ReLU)

    def forward(self, data, device):
        # Extract graph data and move to specified device
        graph_data = data["graph"].to(device)
        if graph_data.dim() != 3:
            # If graph data is not 3D (B, N, N), add an extra dimension
            graph_data = graph_data.unsqueeze(0)

        # Extract vertex features and move to device
        vertices_feature_x = data["vertices_feature_x"].to(device)
        if vertices_feature_x.dim() != 3:
            # If vertex features are not 3D (B, N, F), add an extra dimension
            vertices_feature_x = vertices_feature_x.unsqueeze(0)

        B, N = vertices_feature_x.size(0), vertices_feature_x.size(1)

        # Apply the first graph convolutional layer followed by ReLU activation
        output_1 = self.act(self.conv1(vertices_feature_x, graph_data))

        # Apply the second graph convolutional layer followed by ReLU activation
        output_2 = self.act(self.conv2(output_1, graph_data))

        F = output_2.size(2)  # Get the number of features in the final output
        output_2 = torch.reshape(output_2, (B, N * F))  # Flatten the output (B, N * F)

        # Apply the fully connected layers to the flattened output
        output_2 = self.fc(output_2)

        # Return the final output (after removing extra dimensions)
        return output_2.squeeze()



