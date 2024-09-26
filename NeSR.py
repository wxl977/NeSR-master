import torch
import torch.nn as nn
import torch.nn.functional as F
from recommender import GraphRecommender
from util import OptionConf, next_batch_pairwise
import faiss
from torch_cluster import random_walk

def bpr_loss(user_e, pos_e, neg_e):
    pos_score = torch.mul(user_e, pos_e).sum(dim=1)
    neg_score = torch.mul(user_e, neg_e).sum(dim=1)
    loss = - torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2) / emb.shape[0]
    return emb_loss * reg

def batch_softmax_loss(view1_e, view2_e, temperature):
    v1, v2 = F.normalize(view1_e, dim=1), F.normalize(view2_e, dim=1)
    pos_score = (v1 * v2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(v1, v2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(loss)


class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)


class NeSR(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(NeSR, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['NeSR'])
        self.n_layers = int(args['-n_layer'])  # 3
        self.ssl_temp = float(args['-tau'])  # 0.05
        self.ssl_reg = float(args['-str_reg'])  # 1e-2
        self.sim_reg = float(args['-sim_reg'])  # 1e-2
        self.k = int(args['-num_clusters'])  # 500
        self.walk = int(args['-walk'])  # 5
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)
        self.find_n = 2
        self.eps = 0.01

        sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda()
        self.row_list = sparse_norm_adj._indices()[0]
        self.col_list = sparse_norm_adj._indices()[1]

    def e_step(self):
        user_embeddings = self.model.embedding_dict['user_emb'].detach().cpu().numpy()
        item_embeddings = self.model.embedding_dict['item_emb'].detach().cpu().numpy()
        self.user_sim_neighs, self.user_sim_dis, self.user_label = self.find_sim_nodes(user_embeddings)
        self.item_sim_neighs, self.item_sim_dis, self.item_label = self.find_sim_nodes(item_embeddings)

    def find_sim_nodes(self, x):
        quant = faiss.IndexFlatL2(self.emb_size)
        index = faiss.IndexIVFFlat(quant, self.emb_size, self.k)
        assert not index.is_trained
        index.train(x)
        assert index.is_trained
        index.add(x)
        D, I = index.search(x, self.find_n)
        sim_nodes = torch.Tensor(I).cuda()
        dis_nodes = torch.Tensor(D).cuda()
        labels = index.quantizer.assign(x, 1)
        node_label = torch.tensor(labels).cuda()
        return sim_nodes, dis_nodes, node_label

    def Orien_sim_loss(self, initial_emb, user_idx, item_idx):
        user_emb, item_emb = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        user_1 = self.user_sim_neighs[user_idx][:, 0]  # [batch]
        item_1 = self.item_sim_neighs[item_idx][:, 0]
        user_neigh = self.user_sim_neighs[user_idx][:, 1]  # shape [batch，5]
        item_neigh = self.item_sim_neighs[item_idx][:, 1]  # shape [batch, 5]

        user_idx = torch.Tensor(user_1).type(torch.long).cuda()
        sim_user = torch.Tensor(user_neigh).type(torch.long).cuda()
        item_idx = torch.Tensor(item_1).type(torch.long).cuda()
        sim_item = torch.Tensor(item_neigh).type(torch.long).cuda()

        user_label = self.user_label[user_idx, :].squeeze()
        sim_user_label = self.user_label[sim_user, :].squeeze()
        user_one = F.one_hot(user_label, num_classes=self.k).float()  # user * label
        sim_u_one = F.one_hot(sim_user_label, num_classes=self.k).float()
        pos_u_mask = torch.matmul(user_one, sim_u_one.transpose(0, 1))

        item_label = self.item_label[item_idx, :].squeeze()
        sim_item_label = self.item_label[sim_item, :].squeeze()
        item_one = F.one_hot(item_label, num_classes=self.k).float()
        sim_i_one = F.one_hot(sim_item_label, num_classes=self.k).float()
        pos_i_mask = torch.matmul(item_one, sim_i_one.transpose(0, 1))

        u_loss = self.InfoNCE_label(user_emb[user_idx], user_emb[sim_user], pos_u_mask, self.ssl_temp)
        i_loss = self.InfoNCE_label(item_emb[item_idx], item_emb[sim_item], pos_i_mask, self.ssl_temp)

        knn_nce_loss = self.sim_reg * (u_loss + i_loss)
        return knn_nce_loss

    @staticmethod
    def InfoNCE_label(view1, view2, pos_mask, temperature, b_cos=True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        sim = (view1 * view2).sum(dim=-1)
        sim = torch.exp(sim / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        pos_s = ttl_score * pos_mask
        pos = torch.exp(pos_s / temperature).sum(dim=1)
        cl_loss = - torch.log(sim / pos + 10e-6)
        return torch.mean(cl_loss)

    def rand_walk(self, initial_emb, user_idx, item_idx, walk):
        user_idx = torch.unique(torch.Tensor(user_idx).type(torch.long))
        item_idx = torch.unique((torch.Tensor(item_idx) + self.data.user_num).type(torch.long))
        idx = torch.concat([user_idx, item_idx]).cuda()
        walk_res = random_walk(self.row_list, self.col_list, idx, walk_length=walk)
        idx_rw_emb = 0
        for i in range(walk + 1):
            if i == 0:
                idx_emb = initial_emb[walk_res[:, i]]
            idx_rw_emb += initial_emb[walk_res[:, i]]
        idx_u_len = len(user_idx)
        target_u = idx_emb[:idx_u_len]
        target_i = idx_emb[idx_u_len:]
        idx_rw_emb /= (walk + 1)
        walk_u_neig = idx_rw_emb[:idx_u_len]
        walk_i_neig = idx_rw_emb[idx_u_len:]
        u_loss = batch_softmax_loss(target_u, walk_u_neig, self.ssl_temp)
        i_loss = batch_softmax_loss(target_i, walk_i_neig, self.ssl_temp)
        ns_loss = self.ssl_reg * (u_loss + i_loss)
        return ns_loss

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            if epoch >= 20:
                self.e_step()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                rec_user_emb, rec_item_emb, emb_list = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                initial_emb = emb_list[0]
                ns_rw_loss = self.rand_walk(initial_emb, user_idx, pos_idx, self.walk)
                warm_up_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,
                                                      neg_item_emb) / self.batch_size + ns_rw_loss
                if epoch < 20:  # warm_up
                    optimizer.zero_grad()
                    warm_up_loss.backward()
                    optimizer.step()
                    if n % 100 == 0 and n > 0:
                        print('training:', epoch + 1, 'batch', n, 'rec_loss:%.6f' % (rec_loss.item()),
                              'ssl_loss:%.6f' % (ns_rw_loss.item()))
                else:
                    sim_loss = self.Orien_sim_loss(initial_emb, user_idx, pos_idx)
                    batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,
                                                        neg_item_emb) / self.batch_size + ns_rw_loss + sim_loss
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    if n % 100 == 0 and n > 0:
                        print('training:', epoch + 1, 'batch', n, 'rec_loss:%.6f' % (rec_loss.item()),
                              'ns_rw_loss:%.6f' % (ns_rw_loss.item()), 'proto_loss:%.6f' % (sim_loss.item()))
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb, _ = model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _ = self.model()


    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_  # require_grad=True
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        lgcn_all_embeddings = torch.stack(all_embeddings, dim=1)
        lgcn_all_embeddings = torch.mean(lgcn_all_embeddings, dim=1)
        user_all_embeddings = lgcn_all_embeddings[:self.data.user_num]
        item_all_embeddings = lgcn_all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings, all_embeddings