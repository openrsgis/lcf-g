import pandas as pd
from torch.utils.data import Dataset
import dill
import torch
import numpy as np

def get_feature_vertices_data():
    with open('data/adjacencies.pkl', 'rb') as f:
        adjacencies = dill.load(f)
    with open('data/feature_vertices_data_list.pkl', 'rb') as f:
        feature_vertices_data_list = dill.load(f)
    with open('data/labels.pkl', 'rb') as f:
        labels = dill.load(f)
    return {"data":feature_vertices_data_list,"labels":labels,"adjacencies":adjacencies}  # list[N, D]

class LoadData(Dataset):
    def __init__(self,num_nodes,train_mode,graph_type):
        self.train_mode=train_mode
        self.num_nodes=num_nodes
        self.graph_type=graph_type
        if self.train_mode == "train":
            csv_file = "path/to/train.csv"
            self.data_locations = pd.read_csv(csv_file)
        elif self.train_mode == "val":
            csv_file = "path/to/val.csv"
            self.data_locations = pd.read_csv(csv_file)
        elif self.train_mode == "test":
            csv_file = "ppath/to/test.csv"
            self.data_locations = pd.read_csv(csv_file)
        self.feature_vertices_norm,self.feature_vertices_data,self.labels,self.adjacencies=self.pre_process_data(data=get_feature_vertices_data(),norm_dim=1,train_mode=train_mode)
    def __len__(self):
        """
        Represents the length of the data set
        :return: length of dataset (number of samples).
        """
        if self.train_mode=="train":
            return len(self.data_locations)
        elif self.train_mode=="val":
            return len(self.data_locations)
        elif self.train_mode=="test":
            return len(self.data_locations)
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))
    def __getitem__(self,index):
        """
        Function is to take each sample (x, y), index = [0, L1-1] This is determined by the length of the data set.
        :param index: int, range between [0,length-1].
        :return:
            graph: torch.tensor, [N,N].
            data_x: torch.tensor, [N, N, D].
            data_y: torch.tensor, [N, 1, D].
        """

        idx=self.data_locations.iloc[index, 0]
        data_x,data_y=LoadData.get_data(self.labels,self.feature_vertices_data,idx)

        data_x=LoadData.to_tensor(data_x)
        data_y=LoadData.to_tensor(data_y)

        graph=self.adjacencies[idx]
        # maxnum_vertices=1024
        maxnum_vertices=2048

        A = np.zeros([int(maxnum_vertices), int(maxnum_vertices)])  # Construct an all 0 adjacency matrix.

        nnz = np.nonzero(graph)

        if self.graph_type == "connect":
            for i, j in zip(nnz[0], nnz[1]):
                A[i][j] = 1
        elif self.graph_type == "distance":
            for i, j in zip(nnz[0], nnz[1]):
                if graph[i, j] != 0:
                    A[i][j] = 1. / graph[i, j]
                else:
                    A[i][j] = float("100000")
        return {"graph":LoadData.to_tensor(A),"vertices_feature_x":data_x,"vertices_feature_y":data_y}

    @staticmethod
    def get_data(labels, data, idx,max_nodes=2048):
        """
        Divide data samples according to proportion
        :param data: np.array, normalized feature vertices data.
        :param train_mode: str, ["train", "test"].
        :return:
            data_x: np.array, [N,H,D].
            data_y: np.array, [N,D].
        """

        feature_vertices = data
        feature_vertice = feature_vertices[idx]

        if (feature_vertice.shape[0] > max_nodes):
            print("feature_vertice.shape", feature_vertice.shape)

        data_x = np.pad(feature_vertice,
                        ((0, 2048 - feature_vertice.shape[0]), (0, 0)),
                        'constant', constant_values=(0))
        data_x = data_x.astype(np.float32)
        data_y = labels[idx]
        data_y = float(data_y)
        data_x = LoadData.to_tensor(data_x)
        data_y = LoadData.to_tensor(data_y)
        return data_x, data_y
    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)

    @staticmethod
    def recover_data(max_data, min_data, data):  # Used during data recovery, prepared for visualization comparison.
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, normalized data.
        :return:
            recovered_data: np.array, recovered data.
        """
        mid = min_data
        base = max_data - min_data

        recovered_data = data * base + mid

        return recovered_data

    @staticmethod
    def pre_process_data(data, norm_dim, train_mode):
        """
        Preprocessing, normalization
        :param data: np.array, raw feature data
        :param nom_dim: int, Normalized dimensions, that is, in what dimension, in this case dim=1 time dimension
        :return:
            norm_base: list, [max_data, min_data], This is a normalized basis.
            norm_data: np.array, normalized feature data.
        """
        norm_base = LoadData.normalize_base(data["data"], norm_dim, train_mode)
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data["data"])

        return norm_base, norm_data, data["labels"], data["adjacencies"]

    @staticmethod
    def normalize_base(data, norm_dim, train_mode):
        """
        Compute the normalized basis
        :param data: np.array, raw vertex feature data
        :param norm_dim: int, normalization dimension.
        Normalized dimensions, that is, in what dimension, in this case dim=1 time dimension
        :return:
            max_data: np.array
            min_data: np.array
        """
        concatenate_Geometry = np.concatenate(data, axis=0)
        concatenate_Geometry = np.abs(concatenate_Geometry)
        max_data = concatenate_Geometry.max(axis=0)
        min_data = concatenate_Geometry.min(axis=0)
        if train_mode == "train":
            file = "path/to/conf_point.txt"
            conc = np.vstack((max_data, min_data))
            np.savetxt(file, conc, fmt='%.18f')
        else:
            file = "path/to/conf_point.txt"
            conc = np.loadtxt(file)
            max_data, min_data = conc[0, :], conc[1, :]
        return max_data, min_data
    @staticmethod
    def normalize_data(max_data,min_data,data):
        """
         Calculate normalized traffic data, using the max-min normalization method
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, original vertex feature data without normalization.
        :return:
            np.array, normalized vertex feature data.
        """
        mid = min_data
        base = max_data-min_data
        return data