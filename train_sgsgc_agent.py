import os

import torch
import torch.nn as nn
from coreset import Random
from utils import *
import numpy as np
from models.SWGCN import SignedGraphConvolutionalNetwork as SWGCN
from models.SNEA import SNEA
from models.SSSNET import SSSNET_link_prediction as SSSNET
from models.parametrized_adj import PGE
import sys
from tqdm import tqdm
sys.path.extend('data/lr')

loss_func = torch.nn.BCELoss()


class SGSGC:

    def __init__(self, data, args, **kwargs):
        self.data = data
        self.args = args
        self.device = args.device

        n, d = int(data.nnodes * args.reduction_rate), args.xdim
        self.nnodes_syn = n

        self.feat = data.feat.to(args.device)
        self.edges_pos_train, self.edges_neg_train = data.edges_pos_train.to(args.device), data.edges_neg_train.to(
            args.device)
        self.edges_pos_test, self.edges_neg_test = data.edges_pos_test.to(args.device), data.edges_neg_test.to(
            args.device)
        self.labels_train = torch.tensor(
            [1] * self.edges_pos_train.shape[1] + [0] * self.edges_neg_train.shape[1]).type(torch.float).to(args.device)
        self.labels_test = torch.tensor([1] * self.edges_pos_test.shape[1] + [0] * self.edges_neg_test.shape[1])

        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(self.device))
        self.pge = PGE(nfeat=d, nnodes=n, device=self.device, args=args).to(self.device)

        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        print('adj_syn:', (n,n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        if self.args.initialize == False:
            self.feat_syn.data.copy_(torch.randn_like(self.feat_syn))
        else:
            agent = Random(self.data, self.args, device=self.args.device)
            idx_selected = agent.select(None)
            feat_syn = self.feat[idx_selected,:]
            self.feat_syn.data.copy_(feat_syn)

    def test_with_val(self, verbose=True):
        args = self.args
        feat = self.feat
        feat_syn, pge = self.feat_syn.detach(), self.pge
        adj_syn = pge.inference(feat_syn)
        adj_syn[(adj_syn < args.thred) & (adj_syn > -args.thred)] = 0.0
        edge_index_pos_syn, edge_index_neg_syn, edge_attr_pos_syn, edge_attr_neg_syn, \
        labels_syn = self.adj_to_index(adj_syn)
        ps, ns = edge_index_pos_syn.shape[1], edge_index_neg_syn.shape[1]

        edges_pos_test, edges_neg_test = self.edges_pos_test, self.edges_neg_test
        labels_test = self.labels_test
        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        model = eval(args.model)(args, args.xdim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_model, weight_decay=args.weight_decay)

        model.train()
        for epoch in range(600):
            # start_time = time.time()
            optimizer.zero_grad()
            out, _ = model(edge_index_pos_syn, edge_index_neg_syn, feat_syn, edge_attr_pos_syn, edge_attr_neg_syn)
            loss = loss_func(out, labels_syn)
            loss.backward()
            optimizer.step()

        model.eval()
        out, _ = model(edges_pos_test, edges_neg_test, feat)
        re = score(out, labels_test)

        pos_rate = edge_index_pos_syn.shape[1] / (edge_index_pos_syn.shape[1] + edge_index_neg_syn.shape[1])
        print('ps:{}, ns:{}'.format(ps, ns))
        print('test auc:{:.4f} acc_pos:{:.4f} acc_neg:{:.4f} pr:{:.4f}'.format(re[0], re[1], re[2], re[3]))
        return re + [pos_rate]

    def train(self, verbose=True):
        global loss, ps, ns, adj_syn
        args = self.args

        feat_syn, pge = self.feat_syn, self.pge

        edges_pos, edges_neg = self.edges_pos_train, self.edges_neg_train
        #adj = index_to_adj(edges_pos, edges_neg, self.data.nnodes).to(self.device)
        labels_train = self.labels_train
        pr, nr = edges_pos.shape[1], edges_neg.shape[1]
        feat = self.feat

        coeff = [1/3, 2/3]
        outer_loop, inner_loop = args.outer, args.inner

        for it in tqdm(range(args.epochs + 1)):
            model = eval(args.model)(args, args.xdim).to(self.device)
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)

            model.train()
            loss_avg = 0
            for ol in range(outer_loop):
                adj_syn = pge(self.feat_syn)
                edge_index_pos_syn, edge_index_neg_syn, edge_attr_pos_syn, edge_attr_neg_syn, \
                labels_syn = self.adj_to_index(adj_syn)
                ps, ns = edge_index_pos_syn.shape[1], edge_index_neg_syn.shape[1]

                out_real, z_real = model(edges_pos, edges_neg, feat)
                lpr = loss_func(out_real[labels_train==1], labels_train[labels_train==1])
                lnr = loss_func(out_real[labels_train==0], labels_train[labels_train==0])
                loss_real_list = [lpr, lnr]

                out_syn, z_syn = model(edge_index_pos_syn, edge_index_neg_syn, self.feat_syn,
                                       edge_attr_pos_syn,
                                       edge_attr_neg_syn)
                lps = loss_func(out_syn[:ps], labels_syn[:ps])
                lns = loss_func(out_syn[ps:], labels_syn[ps:])
                loss_syn_list = [lps, lns]

                loss_grad = torch.tensor(0.0).to(self.device)
                for i in range(2):
                    lr = loss_real_list[i]
                    gw_real = torch.autograd.grad(lr, model_parameters, create_graph=True, allow_unused=True)
                    gw_real = list((_.detach().clone() for _ in gw_real if _ is not None))

                    ls = loss_syn_list[i]
                    gw_syn = torch.autograd.grad(ls, model_parameters, create_graph=True, allow_unused=True)
                    gw_syn = list((_ for _ in gw_syn if _ is not None))
                    # coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss_grad += coeff[i] * gradient_loss(gw_syn, gw_real, device=self.device)

                loss_cl, loss_smooth =  self.regularization(z_real, z_syn, adj_syn)

                loss = loss_grad + self.args.alpha * loss_cl + self.args.beta * loss_smooth
                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                if it % 50 < 20:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()


                feat_syn_inner = self.feat_syn.detach()
                adj_syn_inner = self.pge.inference(feat_syn_inner)
                self.adj_syn = adj_syn_inner
                if ol == outer_loop - 1:
                    # print('loss_reg:', loss_reg.item())
                    # print('Gradient matching loss:', loss.item())
                    break

                adj_syn_inner[(adj_syn_inner < args.thred) & (adj_syn_inner > -args.thred)] = 0
                edge_index_pos_syn_inner, edge_index_neg_syn_innder, edge_attr_pos_syn_inner, \
                edge_attr_neg_syn_inner, labels_syn_inner = self.adj_to_index(adj_syn_inner)

                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    out, _ = model(edge_index_pos_syn_inner, edge_index_neg_syn_innder, feat_syn_inner,
                                   edge_attr_pos_syn_inner, edge_attr_neg_syn_inner)
                    loss_syn_inner = loss_func(out, labels_syn_inner)
                    loss_syn_inner.backward()
                    # print('inner epoch [{}/{}] model loss:{:.4f}'.format(j+1, inner_loop, loss_syn_inner.item()))
                    optimizer_model.step()

            #print('Epoch {}, loss:{:.4f} grad:{:.4f} gen:{:.4f} smooth:{:.4f} ps:{}, ns:{}'.format(it, loss.item(),loss_grad.item(), loss_cl.item(), loss_smooth.item(), ps, ns))

            eval_epochs = [100 * (i + 1) for i in range(100)]

            if it in eval_epochs:
                re = self.test_with_val()
                #path = '{}train/auc{:.3f}_pos{:.3f}_neg{:.3f}/'.format(args.path, re[0], re[1], re[2])
                #if os.path.exists(path) == False:
                #    os.makedirs(path)
                #torch.save(self.adj_syn.detach(), path + 'adj_syn.pt')
                #torch.save(self.feat_syn.detach(), path + 'feat_syn.pt')

    def adj_to_index(self, adj):
        #adj[(adj < self.args.thred) & (adj > -self.args.thred)] = 0
        edge_index_pos, edge_index_neg = torch.nonzero(adj > 0).T, torch.nonzero(adj < 0).T
        #adj = torch.abs(adj) * 2 - 1
        edge_attr_pos = adj[edge_index_pos[0], edge_index_pos[1]]
        edge_attr_neg = adj[edge_index_neg[0], edge_index_neg[1]]
        ps, ns = edge_index_pos.shape[1], edge_index_neg.shape[1]
        labels = np.array([1] * ps + [0] * ns)
        labels = torch.from_numpy(labels).type(torch.float).to(self.device)

        return edge_index_pos, edge_index_neg, edge_attr_pos, edge_attr_neg, labels

    def regularization(self,z_real, z_syn, adj_):
        index = [[], [], []]
        adj_syn = adj_.to('cpu')
        a = np.arange(adj_syn.shape[0])
        for i in range(z_syn.shape[0]):
            pi, ni = a[adj_syn[i]>0], a[adj_syn[i]<0]
            if len(pi)==0 or len(ni)==0:
                continue
            index[0] = index[0] + [i] * 10
            index[1] = index[1] + list(np.random.choice(pi, 10))
            index[2] = index[2] + list(np.random.choice(ni, 10))

        edges = self.edges_pos_train
        if edges.shape[1] > 2000:
            random_index = np.random.choice(np.arange(edges.shape[1]), 2000, replace=False)
        else:
            random_index = np.arange(edges.shape[1])
        i_c = torch.exp(z_real[edges[0, random_index]] @ z_syn.T)
        i_c = i_c / i_c.sum(1).view(-1, 1)
        i_c[torch.isnan(i_c)] = 0
        j_c = -self.log_sigmoid(z_real[edges[1, random_index]] @ z_syn.T)
        kk = torch.zeros_like(i_c)
        kk.scatter_add_(1, torch.tensor(index[0]).expand(kk.shape[0], -1).to(self.device),
                        (j_c[:, index[1]] - j_c[:, index[2]]).clamp(min=0))
        lp = (i_c * kk).sum(1).mean()

        edges = self.edges_neg_train
        if edges.shape[1] > 2000:
            random_index = np.random.choice(np.arange(edges.shape[1]), 2000, replace=False)
        else:
            random_index = np.arange(edges.shape[1])
        i_c = torch.exp(z_real[edges[0, random_index]] @ z_syn.T)
        i_c = i_c / i_c.sum(1).view(-1, 1)
        i_c[torch.isnan(i_c)] = 0
        j_c = -self.log_sigmoid(z_real[edges[1, random_index]] @ z_syn.T)
        kk = torch.zeros_like(i_c)
        kk.scatter_add_(1, torch.tensor(index[0]).expand(kk.shape[0], -1).to(self.device),
                        (-j_c[:, index[1]] + j_c[:, index[2]]).clamp(min=0))
        ln = (i_c * kk).sum(1).mean()
        loss_gen = (lp + ln) / 2

        loss_smooth = torch.zeros(z_syn.shape[0]).to(self.device)
        loss_smooth.scatter_add_(0, torch.tensor(index[0]).to(self.device),
                                 (-self.log_sigmoid((z_syn[index[0]] * z_syn[index[1]]).sum(1)) + self.log_sigmoid(
                                     (z_syn[index[2]] * z_syn[index[0]]).sum(1))).clamp(min=0))
        # loss_smooth = (z_syn[index[0]] - z_syn[index[1]]).pow(2).sum(1) - (z_syn[index[0]] - z_syn[index[2]]).pow(2).sum(1)
        loss_smooth = loss_smooth.mean()

        return loss_gen, loss_smooth

    def log_sigmoid(self, x):
        return torch.log(torch.sigmoid(x))