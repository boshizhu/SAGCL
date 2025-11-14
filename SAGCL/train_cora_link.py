from time import perf_counter as t
from evaluate import get_roc_score
import random
import time
import torch
from torch_geometric.utils import is_undirected, to_undirected
import os.path as osp
import torch.nn.functional as F
import torch.nn as nn
from colorama import Fore
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.datasets import Planetoid, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from scipy.sparse import coo_matrix, csr_matrix
from data_unit.utils import blind_other_gpus, row_normalize, sparse_mx_to_torch_sparse_tensor,normalize_graph
from models import LogReg, SAGCL
from torch_geometric.utils import degree
import os
import argparse
from datasets import do_edge_split_direct, get_dataset
from torch_sparse import SparseTensor
from sklearn.cluster import KMeans
from ruamel.yaml import YAML
from termcolor import cprint
from evaluate import mask_test_edges
from eval import test
from evaluate import clustering_metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
def get_args_key(args):
    return "-".join([args.model_name, args.dataset_name, args.custom_key])
def link_decoder(h, edge):
    src_x = h[edge[0]]
    dst_x = h[edge[1]]

    x = (src_x * dst_x).sum(1)

    return x
def get_args(model_name, dataset_class, dataset_name, custom_key="", yaml_path=None) -> argparse.Namespace:
    yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    custom_key = custom_key.split("+")[0]
    parser = argparse.ArgumentParser(description='Parser for Simple Unsupervised Graph Representation Learning')
    # Basics
    parser.add_argument("--num-gpus-total", default=0, type=int)
    parser.add_argument("--num-gpus-to-use", default=0, type=int)
    parser.add_argument("--black-list", default=None, type=int, nargs="+")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--model_name", default=model_name)
    parser.add_argument("--custom_key", default=custom_key)
    parser.add_argument("--save_model", default=False)
    parser.add_argument("--seed", default=0)
    # Dataset
    parser.add_argument('--data-root', default="~/graph-data", metavar='DIR', help='path to dataset')
    parser.add_argument("--dataset-class", default=dataset_class)
    parser.add_argument("--dataset-name", default=dataset_name)
    # Pretrain
    parser.add_argument("--pretrain", default=False, type=bool)
    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=1024, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr2', '--learning-rate2', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate2', dest='lr2')
    parser.add_argument("--use-bn", default=False, type=bool)
    parser.add_argument("--perf-task-for-val", default="Node", type=str)  # Node or Link
    parser.add_argument('--w_loss1', type=float, default=1, help='')
    parser.add_argument('--w_loss2', type=float, default=1, help='')
    parser.add_argument('--w_loss3', type=float, default=1, help='')
    parser.add_argument('--w_loss_c', type=float, default=1, help='')
    parser.add_argument('--margin1', type=float, default=0.8, help='')
    parser.add_argument('--margin2', type=float, default=0.2, help='')
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--Init', type=str,
                        default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--rank', type=int, default=3)
    parser.add_argument('--patience', type=int, default=50)
    # Experiment specific parameters loaded from .yamls
    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset_name or args.dataset_class, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")
    # Update params from .yamls
    args = parser.parse_args()
    return args

def pprint_args(_args: argparse.Namespace):
    cprint("Args PPRINT: {}".format(get_args_key(_args)), "yellow")
    for k, v in sorted(_args.__dict__.items()):
        print("\t- {}: {}".format(k, v))

# def load_graph_dataset(dataset_name, device, re_split=False):
#     graph_data = torch.load(f"D:\LLM\LLMNodeBed-main\LLMNodeBed-main\datasets\citeseer.pt", weights_only=False).to(device)
#     # Alternative
#     # graph_data.edge_index = to_undirected(graph_data.edge_index) if dataset_name in ["citeseer", "arxiv"] else graph_data.edge_index
#     graph_data.edge_index = to_undirected(graph_data.edge_index)
# def load_graph_dataset_for_gnn(dataset_name, device, re_split=False):
#     graph_data = load_graph_dataset(dataset_name, device, re_split)

def get_dataset(args, dataset_kwargs):
    #args.dataset="CiteSeer"
    data = torch.load('D:\LLM\LLMNodeBed-main\LLMNodeBed-main\datasets\cora.pt').cpu()
    # data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    i = torch.LongTensor([data.edge_index[0].numpy(), data.edge_index[1].numpy()])
    v = torch.FloatTensor(torch.ones([data.num_edges]))
    A_sp = torch.sparse.FloatTensor(i, v, torch.Size([data.num_nodes, data.num_nodes]))
    A = A_sp.to_dense()
    I = torch.eye(A.shape[1]).to(A.device)
    A_I = A + I
    # A_nomal = normalize_graph(A)
    A_I_nomal = normalize_graph(A_I)
    A_I_nomal = A_I_nomal.to_sparse()

    lable = data.y
    nb_feature = data.num_features
    nb_classes = int(lable.max() - lable.min()) + 1
    nb_nodes = data.num_nodes
    data.x = torch.FloatTensor(data.x)
    eps = 2.2204e-16
    norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
    data.x = data.x.div(norm.expand_as(data.x))
    adj_1 = csr_matrix(
        (np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
        shape=(data.num_nodes, data.num_nodes))

    # data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    data.x = torch.FloatTensor(data.x)
    eps = 2.2204e-16
    norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
    data.x = data.x.div(norm.expand_as(data.x))
    adj = coo_matrix(
        (np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
        shape=(data.num_nodes, data.num_nodes))
    nb_nodes = data.num_nodes
    I = coo_matrix((np.ones(nb_nodes), (np.arange(0, nb_nodes, 1), np.arange(0, nb_nodes, 1))),
                   shape=(nb_nodes, nb_nodes))
    adj_I = adj + I  # coo_matrix(sp.eye(adj.shape[0]))
    adj_I = row_normalize(adj_I)
    A_I_nomal = sparse_mx_to_torch_sparse_tensor(adj_I).float()
    lable = data.y
    nb_feature = data.num_features
    nb_classes = int(lable.max() - lable.min()) + 1

    return data, [A_I_nomal,adj_1], [data.x], [lable, nb_feature, nb_classes, nb_nodes]


def run_SUGRL(args, i):
    # ===================================================#
    torch.manual_seed(args.seed+i)
    torch.cuda.manual_seed(args.seed+i)
    torch.cuda.manual_seed_all(args.seed+i)
    np.random.seed(args.seed+i)
    random.seed(args.seed+i)
    # ===================================================#
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    running_device = "cuda:0"
    # ===================================================#
    cprint("## Loading Dataset ##", "yellow")
    dataset_kwargs = {}
    data, adj_list, x_list, nb_list = get_dataset(args, dataset_kwargs)
    split_edge = do_edge_split_direct(data)
    data.edge_index = to_undirected(split_edge['train']['edge'].t())
    edge_index = data.edge_index
    adj = SparseTensor.from_edge_index(edge_index).t()
    data = data.to(running_device)
    adj = adj.to(running_device)
    lable = nb_list[0]
    nb_feature = nb_list[1]
    nb_classes = nb_list[2]
    nb_nodes = nb_list[3]
    feature_X = x_list[0].to(running_device)
    A_I_nomal = adj_list[0].to(running_device)
    tsne_lab = []
    ylabelsx = []
    for i in range(0, nb_nodes):
        tsne_lab.append(lable[i])
        ylabelsx.append(i)
    tsne_lab = np.array(tsne_lab)
    ylablesx = np.array(ylabelsx)
    adj_1 = adj_list[1]


    ############################################################
    emb_1 = np.load(r"D:\archive\llm4graph\cora_Qwen2.5_emb.npy")
    emb_2 = np.load(r"D:\archive\llm4graph\cora_llama8b_emb.npy")
    emb_3 = np.load(r"D:\archive\llm4graph\cora_qwen7b_emb.npy")
    # C:\Users\admin\PycharmProjects\llm4graph\cora_MiniLM_emb.npy
    # C:\Users\admin\PycharmProjects\llm4graph\cora_qwen7b_emb.npy
    # C:\Users\admin\PycharmProjects\llm4graph\cora_llama8b_emb.npy
    embs = [emb_2, emb_3, emb_1]
    embs = [torch.tensor(emb) for emb in embs]
    embs = [(emb).cuda() for emb in embs]
    ####################################################################


    # adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    # test_edges, test_edges_false = mask_test_edges(adj_1, test_frac=0.2, val_frac=0.2)
    cprint("## Done ##", "yellow")
    # ===================================================#
    model = SAGCL(args,nb_feature, cfg=args.cfg,
                       dropout=args.dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(running_device)
    A_degree = degree(A_I_nomal._indices()[0], nb_nodes, dtype=int).tolist()
    edge_index = A_I_nomal._indices()[1]
    # ===================================================#
    my_margin = args.margin1
    my_margin_2 = my_margin + args.margin2
    margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
    num_neg = args.NN
    lbl_z = torch.tensor([0.]).to(running_device)
    deg_list_2 = []
    deg_list_2.append(0)
    for i in range(nb_nodes):
        deg_list_2.append(deg_list_2[-1] + A_degree[i])
    idx_p_list = []
    for j in range(1, 101):
        random_list = [deg_list_2[i] + j % A_degree[i] for i in range(nb_nodes)]
        idx_p = edge_index[random_list]
        idx_p_list.append(idx_p)

    best_valid = 0.0
    best_epoch = 0
    cnt_wait = 0
    best_result = 0

    predictor = link_decoder
    for current_iter, epoch in enumerate(tqdm(range(args.start_epoch, args.start_epoch + args.epochs + 1))):
        model.train()
        optimiser.zero_grad()
        idx_list = []
        for i in range(num_neg):
            idx_0 = np.random.permutation(nb_nodes)
            idx_list.append(idx_0)

        h_a, h_p, h_p_1 = model(embs, feature_X, A_I_nomal)

        s_p = F.pairwise_distance(h_a, h_p)
        s_p_1 = F.pairwise_distance(h_a, h_p_1)
        s_p_2 = F.pairwise_distance(h_p, h_p_1)

        s_n_list = []
        s_n_list_1 = []
        for h_n in idx_list:
            s_n = F.pairwise_distance(h_a, h_a[h_n])
            s_n_1 = F.pairwise_distance(h_p, h_p[h_n])  # 负样本是GNN
            s_n_list.append(s_n)
            s_n_list_1.append(s_n_1)
        margin_label = -1 * torch.ones_like(s_p)

        loss_mar = 0
        loss_mar_1 = 0
        loss_mar_2 = 0
        mask_margin_N = 0
        for s_n in s_n_list:
            loss_mar += (margin_loss(s_p, s_n, margin_label)).mean()
            loss_mar_1 += (margin_loss(s_p_1, s_n, margin_label)).mean()
            loss_mar_2 += (margin_loss(s_p_2, s_n_1, margin_label)).mean()
            mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
        mask_margin_N = mask_margin_N / num_neg

        loss = loss_mar * args.w_loss1 + loss_mar_1 * args.w_loss2 + loss_mar_2 * args.w_loss3 + mask_margin_N
        loss.backward()
        optimiser.step()
        result = test(h_p,feature_X, A_I_nomal,model, predictor, data, adj, split_edge, 128)
        valid_hits = result['AUC'][1]

        if valid_hits > best_valid:
            best_valid = valid_hits
            best_epoch = epoch
            best_result = result
            cnt_wait = 0
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    test_auc = best_result['AUC'][2] * 100
    test_ap = best_result['AP'][2] * 100
    print(f'Final result: Epoch:{best_epoch}, auc: {test_auc:.4f}, ap:{test_ap:.4f}')
    return test_auc, test_ap





if __name__ == '__main__':
    main_args = get_args(
        model_name="SUGRL",  # GCN SUGRL
        dataset_class="Planetoid",
        # Planetoid,MyAmazon
        dataset_name="Cora",  # Cora, CiteSeer, PubMed, Photo, Computers
        custom_key="link",  # classification, link, clu
    )

    pprint_args(main_args)
    for i in range(1):
        #main_args.margin2 = 0.1 + 0.1 * i
        for j in range(1):
            #main_args.margin1=0.1+0.1*j
            auc = []
            ap = []
            for z in range(1):
                # if len(main_args.black_list) == main_args.num_gpus_total:
                #     alloc_gpu = [None]
                #     cprint("Use CPU", "yellow")
                # else:
                #     alloc_gpu = blind_other_gpus(num_gpus_total=main_args.num_gpus_total,
                #                                  num_gpus_to_use=main_args.num_gpus_to_use,
                #                                  black_list=main_args.black_list)
                #     if not alloc_gpu:
                #         alloc_gpu = [int(np.random.choice([g for g in range(main_args.num_gpus_total)
                #                                            if g not in main_args.black_list], 1))]
                #     cprint("Use GPU the ID of which is {}".format(alloc_gpu), "yellow")

                t0 = time.perf_counter()
                test_auc,test_ap=run_SUGRL(main_args,z)
                auc.append(test_auc)
                ap.append(test_ap)
                cprint("Done")
            auc_mean, auc_var, ap_mean, ap_var = np.mean(auc), np.var(auc), np.mean(ap), np.var(ap)
            print(f"AUC 均值: {auc_mean}, AUC 方差: {auc_var}")
            print(f"AP 均值: {ap_mean}, AP 方差: {ap_var}")
            with open('cora_link_7.13.txt', 'a') as f:
                f.write("auc:"+str(auc_mean)+"+"+str(auc_var) + '\n')
                f.write("ap:" + str(ap_mean) + "+" + str(ap_var) + '\n')
                f.write(str("margin1:") + str(main_args.margin1) + '\n')
                f.write(str("margin2:") + str(main_args.margin2) + '\n')
