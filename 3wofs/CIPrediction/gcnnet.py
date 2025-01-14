import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self,in_c,hid_c,out_c):
        super(GCN,self).__init__()
        self.linear_1=nn.Linear(in_c,hid_c)
        self.linear_2=nn.Linear(hid_c,out_c)
        self.fc = nn.Sequential(
            nn.Linear(512*out_c, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 1)
        )  # Additional fully connected layer

        self.act=nn.ReLU()

    def forward(self,data,device):

        graph_data = data["graph"].to(device)
        if graph_data.dim()!=3:
            graph_data=graph_data.unsqueeze(0)
        graph_data=GCN.process_graph(graph_data)

        vertices_feature_x=data["vertices_feature_x"].to(device)
        if vertices_feature_x.dim()!=3:
            vertices_feature_x=vertices_feature_x.unsqueeze(0)

        B,N=vertices_feature_x.size(0),vertices_feature_x.size(1)

        output_1=self.linear_1(vertices_feature_x)

        output_1=self.act(torch.matmul(graph_data,output_1))

        output_2=self.linear_2(output_1)
        output_2=self.act(torch.matmul(graph_data,output_2))
        F = output_2.size(2)

        output_2=torch.reshape(output_2,((B,N*F)))
        output_2 = self.fc(output_2)  # Flatten the output and apply the fully connected layer
        return output_2.squeeze()



    @staticmethod
    def process_graph(graph_data):

        b=graph_data.size(0)
        for i in range(0,b):
            sub_graph=graph_data[i]
            N = sub_graph.size(0)
            matrix_i=torch.eye(N,dtype=torch.float,device=sub_graph.device)#unit matrix
            sub_graph+=matrix_i
            degree_matrix=torch.sum(sub_graph,dim=1,keepdim=False)
            degree_matrix=degree_matrix.pow(-1)
            degree_matrix[degree_matrix == float("inf")] = 0.
            degree_matrix=torch.diag(degree_matrix)
            A=torch.mm(degree_matrix, sub_graph)
            graph_data[i]=A
        return graph_data

