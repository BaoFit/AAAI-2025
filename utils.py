import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def score(out, target):
    predictions = out.detach().to('cpu').numpy()
    auc = roc_auc_score(target, predictions)
    pred = [1 if p>0.5 else 0 for p in predictions]
    pred = np.array(pred)
    #f1_pos = f1_score(target, pred)
    #f1_neg = f1_score(target, pred, pos_label=0)
    acc_pos = accuracy_score(target[target==1], pred[target==1])
    acc_neg = accuracy_score(target[target==0], pred[target==0])

    pos_ratio = sum(pred)/len(pred)

    return [auc, acc_pos, acc_neg, pos_ratio]

loss_func = torch.nn.L1Loss()
def pre_loss(model, z_real, z_syn, pos_edges, neg_edges, w, gamma, device):
    z_real_ = w.T @ z_syn
    pe, ne = pos_edges.shape[1], neg_edges.shape[1]
    labels = np.array([1] * pe + [0] * ne)
    labels = torch.from_numpy(labels).type(torch.float).to(device)
    out = model.cal_predictions(pos_edges, neg_edges, z_real)
    out_ = model.cal_predictions(pos_edges, neg_edges, z_real_)
    return (loss_func(out[:pe], out_[:pe]) + gamma * loss_func(out[pe:], out_[pe:])) * 10

def match_loss(h_real, h_syn, W, device):
    loss = torch.tensor(0.0).to(device)
    W_norm = W / torch.sum(W, dim=0)
    W_norm[torch.isinf(W_norm)] = 0
    for i in range(len(h_syn)):
        hr, hs = h_real[i], h_syn[i]
        hr = torch.matmul(W_norm[:,:,None], hr[:,None,:])
        hr = torch.sum(hr, dim=0)
        loss += torch.mean(torch.sum((hr - hs)**2,dim=1))
    return loss / len(h_syn)

def distribution_loss(h_real, h_syn, device):
    loss = torch.tensor(0.0).to(device)
    for i in range(len(h_syn)):
        hr, hs = h_real[i], h_syn[i]
        loss += torch.sum((torch.mean(hr, dim=0) - torch.mean(hs, dim=0)).pow(2))
        #print(loss.item()*10)
    return loss / len(h_syn)

loss_L2 = torch.nn.MSELoss(reduction='sum')
def convolution_loss(h_real, h_syn, w, device):
    loss = torch.tensor(0.0).to(device)
    for i in range(len(h_real)):
        layer_pos, layer_neg = h_real[i], h_syn[i]
        for j in range(len(layer_pos)):
            hr, hs = layer_pos[j], layer_neg[j]
            loss += loss_L2(w.T @ hs, hr)/hr.shape[0]/2
    return loss / (len(h_real) * len(h_real[0]))

def contrastive_loss(h_real, h_syn, temp):
    real_pos, real_neg = h_real
    syn_pos, syn_neg = h_syn
    real_pos_norm = real_pos.mean(dim=0).view(1, -1)
    real_neg_norm = real_neg.mean(dim=0).view(1,-1)

    pos_score = torch.log(torch.exp(syn_pos @ real_pos_norm.T / temp).sum(1) + 1e-8)
    neg_score = torch.log(torch.exp(syn_pos @ real_neg_norm.T / temp).sum(1) + 1e-8).mean()*10
    score = (-pos_score + neg_score).mean()
    pos_score = torch.log(torch.exp(syn_neg @ real_neg_norm.T / temp).sum(1) + 1e-8)
    neg_score = torch.log(torch.exp(syn_neg @ real_pos_norm.T / temp).sum(1) + 1e-8).mean()*10
    score += (-pos_score + neg_score).mean()

    return score / 10

def reconstruct_adj_loss(adj1, adj2, p):
    loss = torch.pow((adj1 - adj2) * p, 2).sum()
    return loss / adj1.shape[0] / 2

def reconstruct_feat_loss(feat1, feat2):
    loss = torch.pow(feat1 - feat2, 2).sum()
    return loss / feat1.shape[0] / 2

def gradient_loss(gw_syn, gw_real, device, num=256):
    dis = torch.tensor(0.0).to(device)

    for ig in range(len(gw_real)):
        gwr = gw_real[ig]
        gws = gw_syn[ig]
        dis += distance_wb(gwr, gws)
    #print((dis / len(gw_real)).item())
    return dis / len(gw_real)

def distance_wb(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def normalize_adj_tensor(adj):
    adj = adj - torch.diag(torch.diag(adj, 0))
    mx = adj
    rowsum = torch.abs(mx).sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx

def normalize_w(w, epsilon):
    w_sigmoid = torch.sigmoid(w)
    return torch.relu(w_sigmoid / w_sigmoid.sum(0).view(-1, 1) - epsilon)

def sampler_edges(edges_pos, edges_neg, attr_pos, attr_neg, edge_rate, pos_rate, nodes_syn):
    edges_num = int(nodes_syn * (nodes_syn - 1) * edge_rate)
    pos_num = int(edges_num * pos_rate)
    neg_num = edges_num - pos_num
    if pos_num%2 == 1:
        pos_num += 1
    if neg_num%2 == 1:
        neg_num += 1
    _, pos_index = torch.topk(attr_pos, pos_num)
    _, neg_index = torch.topk(attr_neg, neg_num)
    labels_syn = np.array([0] * pos_num + [1] * neg_num + [2] * (
                    2 * (pos_num + neg_num)))
    return edges_pos[:, pos_index], edges_neg[:, neg_index], labels_syn


def plot(data):
    # 设置颜色
    colors = plt.get_cmap('Accent')(range(2))
    sub_titles = ['(a)original graph','(b)condensed graph']
    label_list = ['sign:+', 'sign:-']
    # 创建 2x2 网格图
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # 将子图展平
    axs = axs.flatten()

    line_x = [[data[0][0].shape[0]+1, data[0][0].shape[0]+data[0][1].shape[0]+1],
              [data[1][0].shape[0]+1, data[1][0].shape[0]+data[1][1].shape[0]+1]]
    data = [torch.exp(torch.cat(x)).to('cpu').detach().numpy() for x in data]
    # 循环绘制子图
    for i, value in enumerate(data):
        # 获取当前子图
        ax = axs[i]
        # 绘制散点图
        ax.scatter(np.arange(1, value.shape[0]+1), value[:,0], color=colors[0], label=label_list[0], s=2)
        ax.scatter(np.arange(1, value.shape[0]+1), value[:,1], color=colors[1], label=label_list[1], s=2)
        ax.vlines(line_x[i][0], 0, 1, linestyles='dashed', colors='grey')
        ax.vlines(line_x[i][1], 0, 1, linestyles='dashed', colors='grey')
        # 设置标题和坐标轴标签
        ax.set_xlabel(sub_titles[i])
        ax.set_ylabel('probability')

    # 显示图例
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend( lines, labels, bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout()
    # 显示图形
    plt.show()


def mx_inv(mx, sqrt=False):
    mx_diag = torch.diag(torch.abs(mx).sum(1))
    mxL = mx_diag - mx
    U, D, V = torch.svd(mxL)
    eps = 0.009
    D_min = torch.min(D)
    if D_min < eps:
        D_1 = torch.zeros_like(D)
        D_1[D > D_min] = 1 / D[D > D_min]
    else:
        D_1 = 1 / D
    # D_1 = 1 / D #.clamp(min=0.005)
    if sqrt:
        return U @ D_1.sqrt().diag() @ V.t(), U @ D_1.diag() @ V.t()
    else:
        return U @ D_1.diag() @ V.t()

def mx_tr(mx):
    return mx.diag().sum()


def index_to_adj(ep=None, en=None, n=None):
    adj = torch.zeros(n, n)
    if ep is not None:
        adj[ep[0], ep[1]] = 1
    if en is not None:
        adj[en[0], en[1]] = -1
    return adj