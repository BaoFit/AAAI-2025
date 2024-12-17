import random, argparse, os
from utils import *
from train_sgsgc_agent import SGSGC
from dataset import DataGraphSAINT
import warnings
import datetime as dt
import json, sys
from tqdm import tqdm
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:3', help='cuda:2')
parser.add_argument('--dataset', type=str, default='epinions')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--layers', type=list, default=[32, 32])
parser.add_argument('--lr_adj', type=float, default=1e-4)
parser.add_argument('--lr_feat', type=float, default=1e-4)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--reduction_rate', type=float, default=0.03)
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0.2, help='trade off balance set loss and unbalance set loss')
parser.add_argument('--beta', type=float, default=0.2, help='trade off balance set loss and unbalance set loss')
parser.add_argument('--gamma', type=float, default=0.2, help='trade off balance set loss and unbalance set loss')
parser.add_argument('--inner', type=int, default=15)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--xdim', type=int, default=64, help="Number of SVD feature extraction dimensions.")
parser.add_argument('--lamb', type=float, default=1.0, help='Embedding regularization parameter. Default is 1.0.')
parser.add_argument('--thred', type=float, default=0.5, help='Embedding regularization parameter. Default is 1.0.')
parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
parser.add_argument('--initialize', default=True, type=bool)
parser.add_argument('--loss_way', default='distribution', type=str)
parser.add_argument('--model', default='SWGCN', type=str)
parser.add_argument('--path', default='', type=str)
parser.add_argument('--ep_ratio', type=float, default=0.5, help='control the ratio of direct \
                     edges predict term in the graph.')
parser.add_argument('--sinkhorn_iter', type=int, default=5, help='use sinkhorn iteration to \
                    warm-up the transport plan.')
args = parser.parse_args()


with open('configs.json') as file:
    configs = json.load(file)

args.device = configs[args.dataset]['device']
args.epochs = configs[args.dataset]['epochs']
args.outer = configs[args.dataset]['outer']
args.inner = configs[args.dataset]['inner']
args.lr_adj = configs[args.dataset]['lr_adj']
args.lr_feat = configs[args.dataset]['lr_feat']
args.alpha = configs[args.dataset]['alpha']
#args.reduction_rate = configs[args.dataset]['reduction_rate'][0]
args.beta = configs[args.dataset]['beta']
#
seed_list = [15, 44, 101, 711, 29]
#seed_list = [711, 29]
result_all = []
data = DataGraphSAINT(args.dataset)
#for j in range(len(configs[args.dataset]['reduction_rate'])):
for j in [0]:
    args.reduction_rate = configs[args.dataset]['reduction_rate'][0]
    args.alpha = 1
    args.beta = 1
    #if args.model == 'SNEA':
    #    args.beta = 0.01
    result_5 = []
    for i in range(5):
        args.seed = seed_list[i]
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        now = dt.datetime.now()
        args.path = '.../{}_{}_{}_seed{}/{}_{}_{}_{}/{}_{}_{}_{}_{}/'.format(
                                                                                         args.dataset, args.reduction_rate, args.model, args.seed,
                                                                              args.lr_adj, args.lr_feat, args.alpha, args.beta,
                                                                              now.month,
                                                                              now.day, now.hour, now.minute,
                                                                              now.second)
        print(args)
        agent = SGSGC(data, args)
        agent.train()
        result_10 = []
        for _ in range(10):
            re = agent.test_with_val()
            result_10.append(re[0])
        result_all.append(result_10)
        result_5.append(np.mean(result_10))
        path = '{}test/auc{:.3f}_{:.3f}/'.format(args.path, np.mean(result_10), np.std(result_10))
        if os.path.exists(path) == False:
            os.makedirs(path)
        # torch.save(agent.adj_syn.detach().to('cpu'), path + 'adj_syn.pt')
        torch.save(agent.feat_syn.detach().to('cpu'), path + 'feat_syn.pt')
        torch.save(agent.adj_syn.detach().to('cpu'), path + 'adj_syn.pt')
    print('{} auc:{:.1f}±{:.1f}'.format(args.reduction_rate, np.mean(result_5) * 100, np.std(result_5) * 100))


for j in range(3):
    r = []
    for k in range(5):
        r.append(np.mean(result_all[j * 5 + k]))
    print('{:.1f}±{:.1f}'.format(np.mean(r) * 100, np.std(r) * 100))


