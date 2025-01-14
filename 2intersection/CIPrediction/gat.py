import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self,in_c,out_c):
        super(GraphAttentionLayer, self).__init__()
        self.in_c=in_c
        self.out_c=out_c

        self.F=F.softmax

        self.W=nn.Linear(in_c,out_c,bias=False)#y=W*x
        self.b=nn.Parameter(torch.Tensor(out_c))

        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)

    def forward(self, inputs, graph):
        """
        Forward pass of the model, where graph attention is applied on the input features.

        :param inputs: Input features, shape [B, N, C] where B is batch size, N is the number of nodes, C is the number of channels/features per node.
        :param graph: Graph structure, adjacency matrix of the graph, shape [N, N] where N is the number of nodes.
        :return:
            output feature, shape [B, N, D] where B is the batch size, N is the number of nodes, D is the number of output features.
        """
        # Step 1: Apply a linear transformation (W*h)
        h = self.W(inputs)  # Shape: [B, N, D], linear transformation of the input features

        # Step 2: Compute the correlation (dot product) between nodes' features, representing their relation strength
        # Multiply with the graph adjacency matrix to retain only the connections (edges)
        # h: [B, N, D], h.transpose(1, 2): [B, D, N]
        # Outputs: [B, N, N], for each pair of nodes i, j, compute the inner product of their features, weighted by the adjacency matrix.
        # The result will be zero if nodes i and j are not connected.
        outputs = torch.bmm(h, h.transpose(1, 2)) * graph  # [B, N, D] * [B, D, N] -> [B, N, N]

        # Step 3: Replace zeros with negative infinity in the output (because softmax(-inf) = 0)
        # This step ensures that non-connected nodes do not contribute to the attention mechanism.
        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))  # Replace 0 with -inf for unconnected nodes

        # Step 4: Normalize the attention scores along the second dimension (over the connected nodes)
        # This creates the attention coefficients between nodes, which will later be used for aggregation
        attention = self.F(outputs, dim=2)  # Normalize along the second dimension to compute attention

        # Step 5: Aggregate features using attention coefficients
        # Perform the final attention-based aggregation of node features
        # [B, N, N] * [B, N, D] -> [B, N, D]
        # This step computes the weighted sum of the neighboring nodes' features based on the attention coefficients.
        return torch.bmm(attention, h) + self.b  # Return the aggregated features plus the bias term


class GATSubNet(nn.Module):  # This is the multi-head attention mechanism
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATSubNet, self).__init__()

        # Use a loop to add multiple attention heads, and use nn.ModuleList to create a parallel structure.
        # `in_c` is the input feature dimension, `hid_c` is the hidden feature dimension.
        self.attention_module = nn.ModuleList(
            [GraphAttentionLayer(in_c, hid_c) for _ in range(n_heads)]
        )

        # After obtaining different results from each attention head, we aggregate them with another attention layer
        self.out_att = GraphAttentionLayer(hid_c * n_heads, out_c)  # Output aggregation layer
        self.fc = nn.Sequential(
            nn.Linear(1024 * out_c, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 1)
        )  # Additional fully connected layer for final output prediction
        self.act = nn.LeakyReLU()  # Activation function

    def forward(self, input, graph):
        """
        Forward pass for GATSubNet.

        :param input: Node feature input, shape [B, N, in_c] where B is batch size, N is the number of nodes, in_c is the feature size per node
        :param graph: The graph structure, adjacency matrix, shape [N, N]
        :return: Output feature, shape [B, 1], the final aggregated result after applying multi-head attention and a fully connected layer
        """
        B, N = input.size(0), input.size(1)

        # Apply each attention head and concatenate their outputs along the last dimension (feature dimension)
        # This creates an output with features from all attention heads, shape: [B, N, hid_c * n_heads]
        outputs = torch.cat([attn(input, graph) for attn in self.attention_module], dim=-1)

        # Apply activation function after concatenation
        outputs = self.act(outputs)

        # Apply the output aggregation attention layer
        outputs = self.out_att(outputs, graph)
        outputs = self.act(outputs)

        # Flatten the output and apply the fully connected layers
        F = outputs.size(2)
        outputs = torch.reshape(outputs, ((B, N * F)))  # Flattening the node features
        outputs = self.fc(outputs)  # Apply fully connected layers

        return outputs  # Return the final output after the fully connected layers


class GATNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATNet, self).__init__()

        # Initialize the GATSubNet as a part of the GATNet
        self.subnet = GATSubNet(in_c, hid_c, out_c, n_heads)

    def forward(self, data, device):
        # Extract graph data [B, N, N], adjacency matrix representing graph structure
        graph = data["graph"].to(device)

        # If graph dimensions are not 3, add an extra dimension (e.g., for batch processing)
        if graph.dim() != 3:
            graph = graph.unsqueeze(0)

        # Extract node features [B, N, C]
        vertices_feature_x = data["vertices_feature_x"].to(device)

        # If vertices_feature_x dimensions are not 3, add an extra dimension
        if vertices_feature_x.dim() != 3:
            vertices_feature_x = vertices_feature_x.unsqueeze(0)

        B, N = vertices_feature_x.size(0), vertices_feature_x.size(1)

        """
        The section above flattens the time-dependent features as inputs.
        This approach ignores the temporal continuity, which can be considered rough.
        Another approach could be to use time-specific tensors, such as:
        flow[:, :, 0] ... flow[:, :, T-1], which would give T tensors of shape [B, N, C], i.e., [B, N, C]*T.
        Each tensor would be processed by a separate SubNet, resulting in T separate SubNets.
        This can be achieved by defining `self.subnet = [GATSubNet(...) for _ in range(T)]`
        and using `nn.ModuleList` to handle each subnet individually, similar to handling multiple attention heads.
        """

        # Pass node features and graph through the GATSubNet to get the prediction
        prediction = self.subnet(vertices_feature_x, graph)

        return prediction.squeeze()  # Remove the singleton dimension


if __name__=='__main__':
    x=torch.randn(128,1024,3) # [B,N,T,C]
    graph=torch.randn(128,1024,1024) #[N,N]
    data={"vertices_feature_x":x,"graph":graph}

    device=torch.device("cpu")

    net=GATNet(in_c=3,hid_c=6,out_c=2,n_heads=2)

    y=net(data,device)
    print(y.size())
