import os.path
from os import remove
from re import split

import numpy as np
import scipy.sparse as sp

from collections import defaultdict
import os


# loader dataset
class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir, file, content, op='w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + file, op) as f:
            f.writelines(content)

    @staticmethod
    def load_data_set(file, rec_type):
        if rec_type == 'graph':
            data = []
            with open(file) as f:
                for line in f:
                    items = split(' ', line.strip())
                    user_id = items[0]
                    item_id = items[1]
                    weight = items[2]
                    data.append([user_id, item_id, float(weight)])
            return data

    @staticmethod
    def load_data_set_diy(file, rec_type):
        if rec_type == 'graph':
            data = []
            lines = open(file, 'r').readlines()
            for l in lines:
                tmps = l.strip()
                inters = [i for i in tmps.split(' ')]
                u_id, items_id = inters[0], inters[1:]
                items_id = list(set(items_id))
                for i_id in items_id:
                    data.append([u_id, i_id, float(1)])
        return data

class Data(object):
    def __init__(self, conf, training, test):
        self.config = conf
        self.training_data = training
        self.test_data = test

# process graph --> return norm_adj_matrix
class Graph(object):
    def __init__(self):
        pass

    @staticmethod
    def normlize_graph_mat(adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

# process user-item inter.
class Interaction(Data,Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self, conf, training, test)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.__generate_set()

        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)

        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normlize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()

    def __generate_set(self):
        for entry in self.training_data:
            user, item, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
        for entry in self.test_data:
            user, item, rating = entry
            if user not in self.user or item not in self.item:
                continue
            self.test_set[user][item] = rating
            self.test_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.user_num)), shape=(n_nodes, n_nodes), dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def __create_sparse_interaction_matrix(self):
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)
        return interaction_mat

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())
    
    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]










