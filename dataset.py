import torch, os
from random import shuffle
from scipy import sparse
import numpy as np
import pandas as pd
import sys
import h5py, json
from sklearn.decomposition import TruncatedSVD

class DataGraphSAINT:
    '''datasets used  in GraphSAINT paper'''
    def __init__(self, dataset, pr=None):

        dataset_str = '/data/lr/data/signed_networks/'+dataset+'/'
        '''
        if dataset == 'bitcoin_alpha':
            data = pd.read_csv(dataset_str+'soc-sign-bitcoinalpha.csv').values.tolist()
            edges_pos = np.array([edge[0:2] for edge in data if edge[2] > 0]).T
            edges_neg = np.array([edge[0:2] for edge in data if edge[2] < 0]).T
        elif dataset == 'bitcoin_otc':
            data = pd.read_csv(dataset_str+'soc-sign-bitcoinotc.csv.csv').values.tolist()
            edges_pos = np.array([edge[0:2] for edge in data if edge[2] > 0]).T
            edges_neg = np.array([edge[0:2] for edge in data if edge[2] < 0]).T
        elif dataset == 'wiki':
            SRC, TGT, RES = [], [], []
            with open(dataset_str+'wiki-RfA.txt') as f:
                for line in f:
                    if line[:3] == 'SRC':
                        SRC.append(line[4:])
                    elif line[:3] == 'TGT':
                        TGT.append(line[4:])
                    elif line[:3] == 'RES':
                        RES.append(int(line[4:]))
            un = list(set(SRC+TGT))
            map = {k:v for k,v in zip(un, list(range(len(un))))}
            adj = np.zeros((len(un), len(un)))
            for i in range(len(SRC)):
                adj[map[SRC[i]], map[TGT[i]]] = RES[i]
            edges_pos = np.array(np.where(adj == 1))
            edges_neg = np.array(np.where(adj == -1))
        elif dataset == 'slashdot':
            adj = np.zeros((82144, 82144))
            with open(dataset_str+'soc-sign-Slashdot090221.txt') as f:
                for line in f:
                    if line[:1] == '#':
                        continue
                    l = line.split('\n')
                    l = l[0].split('\t')
                    adj[int(l[0]), int(l[1])] = int(l[2])
            edges_pos = np.array(np.where(adj == 1))
            edges_neg = np.array(np.where(adj == -1))
        elif dataset == 'epinions':
            adj = np.zeros((131828, 131828))
            with open(dataset_str+'soc-sign-epinions.txt') as f:
                for line in f:
                    if line[:1] == '#':
                        continue
                    l = line.split('\n')
                    l = l[0].split('\t')
                    adj[int(l[0]), int(l[1])] = int(l[2])
            edges_pos = np.array(np.where(adj == 1))
            edges_neg = np.array(np.where(adj == -1))
        elif dataset == 'reddit':
            data = pd.read_csv(dataset_str + 'soc-redditHyperlinks-title.tsv', delimiter='\t').values.tolist()
            #data = pd.read_csv(dataset_str + 'soc-redditHyperlinks-body.tsv', delimiter='\t').values.tolist()
            #feat_train = pd.read_csv(dataset_str + 'web-redditEmbeddings-subreddits.csv').values.tolist()
            un = [x[0] for x in data] + [x[1] for x in data]
            un = list(set(un))
            map = {k: v for k, v in zip(un, list(range(len(un))))}
            edges_pos = np.array([[map[edge[0]], map[edge[1]]] for edge in data if edge[4] > 0]).T
            edges_neg = np.array([[map[edge[0]], map[edge[1]]] for edge in data if edge[4] < 0]).T

        else:
            data = h5py.File(dataset_str+dataset+'_rfa.mat')
            data = np.transpose(data['Gwl_ud'])
            data = np.triu(data)
            edges_pos = np.array(np.where(data==1))
            edges_neg = np.array(np.where(data==-1))
        '''

        self.edges_pos_train = torch.load(dataset_str+'train_edges_pos.pt')
        self.edges_neg_train = torch.load(dataset_str+'train_edges_neg.pt')
        self.edges_pos_test = torch.load(dataset_str+'test_edges_pos.pt')
        self.edges_neg_test = torch.load(dataset_str+'test_edges_neg.pt')
        self.feat = torch.load(dataset_str+'feat.pt')
        self.nnodes = self.feat.shape[0]
        npos, nneg = self.edges_pos_train.shape[1]+self.edges_pos_test.shape[1], self.edges_neg_train.shape[1]+self.edges_neg_test.shape[1]
        print('\n---------' + dataset + '  info---------')
        print('nodes:{}, pos edges:{}, neg edges:{}, pos rate:{:.2f}%'.format(self.nnodes, npos, nneg,
                                                                              npos * 100 / (npos+nneg)))
        '''
        nnodes = int(max(edges_pos.max(), edges_neg.max()) + 1)
        npos, nneg = edges_pos.shape[1], edges_neg.shape[1]
        nedges = npos + nneg
        print('\n---------' + dataset + '  info---------')
        print('nodes:{}, pos edges:{}, neg edges:{}, pos rate:{:.2f}%'.format(nnodes, npos, nneg,
                                                                              npos * 100 / nedges))

        pos_edge_index, neg_edge_index = list(range(npos)), list(np.arange(nneg))
        shuffle(pos_edge_index)
        shuffle(neg_edge_index)
        idx_train_pos, idx_test_pos = pos_edge_index[:int(npos * 0.7)], pos_edge_index[int(npos * 0.7):]
        idx_train_neg, idx_test_neg = neg_edge_index[:int(nneg * 0.7)], neg_edge_index[int(nneg * 0.7):]

        idx_train_pos = idx_train_pos + [x+npos for x in idx_train_pos]
        idx_test_pos = idx_test_pos + [x+npos for x in idx_test_pos]
        idx_train_neg = idx_train_neg + [x+nneg for x in idx_train_neg]
        idx_test_neg = idx_test_neg + [x+nneg for x in idx_test_neg]

        edges_pos = np.concatenate([edges_pos, edges_pos[::-1, :]], axis=1)
        edges_neg = np.concatenate([edges_neg, edges_neg[::-1, :]], axis=1)
        feat_train = self.create_spectral_features(edges_pos[:,idx_train_pos], edges_neg[:,idx_train_neg], nnodes)
        torch.save(torch.from_numpy(edges_pos[:,idx_train_pos]).type(torch.long), dataset_str + 'train_edges_pos.pt')
        torch.save(torch.from_numpy(edges_neg[:,idx_train_neg]).type(torch.long), dataset_str + 'train_edges_neg.pt')

        torch.save(torch.from_numpy(edges_pos[:,idx_test_pos]).type(torch.long), dataset_str + 'test_edges_pos.pt')
        torch.save(torch.from_numpy(edges_neg[:,idx_test_neg]).type(torch.long), dataset_str + 'test_edges_neg.pt')

        torch.save(torch.from_numpy(feat_train).float(), dataset_str + 'feat.pt')
        '''


    def create_spectral_features(self, p_edges, n_edges, node_count, re=False, xdim=64, svd_ites=30, seed=0):
        #p_edges = np.concatenate([positive_edges,positive_edges[::-1,:]],axis=1)
        #n_edges = np.concatenate([negative_edges,negative_edges[::-1,:]],axis=1)
        train_edges = np.concatenate([p_edges,n_edges],axis=1)
        index_1 = train_edges[0].astype(int)
        index_2 = train_edges[1].astype(int)
        values = np.array([1.0] * p_edges.shape[1] + [-1.0] * n_edges.shape[1]).astype(np.float32)
        shaping = (int(node_count), int(node_count))
        signed_A = sparse.csr_matrix(sparse.coo_matrix((values, (index_1, index_2)),
                                                       shape=shaping))

        svd = TruncatedSVD(n_components=xdim,
                           n_iter=svd_ites,
                           random_state=seed)
        svd.fit(signed_A)
        X = svd.components_.T
        if re:
            return X, torch.sparse_coo_tensor(train_edges, values, size=shaping)
        return X

#data = DataGraphSAINT('reddit')


