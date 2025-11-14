import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
from torch.nn import Parameter,Linear, ModuleList
from torch.nn import TransformerEncoder, TransformerEncoderLayer
def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)

class SAGCL(nn.Module):
    def __init__(self,args, n_in ,cfg = None, dropout = 0.2):
        super(SAGCL, self).__init__()
        self.MLP = make_mlplayers(n_in, cfg)
        self.act = nn.ReLU()
        self.dropout = dropout
        self.dim=args.dim
        self.A = None
        self.sparse = True
        self.target_dim = self.dim
        self.linear_projections = []
        self.transformer_encoder = None
        self.nhead = 8
        self.num_layers = 2

        self.predict=nn.Linear(self.dim*3,self.dim)
        for m in self.modules():
            self.weights_init(m)
        ####################################################节点路由参数
        self.K = args.K
        self.Init = args.Init
        self.alpha = args.alpha

        if args.Init == 'PPR':
            # PPR-like
            TEMP = args.alpha * (1 - args.alpha) ** np.arange(args.K + 1)
            TEMP[-1] = (1 - args.alpha) ** args.K
            TEMP = torch.tensor([TEMP for i in range(args.rank)])
        elif args.Init == 'Random':
            # Random
            bound = np.sqrt(3 / (args.K + 1))
            TEMP = np.random.uniform(-bound, bound, args.KK + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
            TEMP = np.array([TEMP for i in range(args.rank)])
        elif args.Init== 'Fix':
            TEMP = np.ones(args.K + 1)
            TEMP = np.array([TEMP for i in range(args.rank)])
        elif args.Init == 'Mine':
            TEMP = []
            para = torch.ones([args.rank, args.K + 1])
            TEMP = torch.nn.init.xavier_normal_(para)
        elif args.Init == 'Mine_PPR':
            TEMP = args.alpha * (1 - args.alpha) ** np.arange(args.K + 1)  # 创建一个数组 其中每个元素的值根据公式 alpha*(1-alpha)**i 计算得出
            TEMP[-1] = (1 - args.alpha) ** args.K
            TEMP = torch.tensor(
                np.array([TEMP] * args.rank))  # 将 NumPy 数组 TEMP 转换为 PyTorch 张量，并使用 rank 参数指定张量的复制次数，以匹配模型中 gamma 参数的形状。

        self.gamma = Parameter(TEMP.float())

        #         self.proj = Linear(num_classes, rank)
        proj_list = []
        for _ in range(args.K + 1):
            proj_list.append(Linear(args.dim, args.rank))
        self.proj_list = ModuleList(proj_list)
        self.rank = args.rank
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, embs,seq_a, adj=None):

        embs=self.project_embeddings(embs)

        embed1, embed2,embed3=embs
        h_0 = torch.tanh(self.proj_list[0](embed1))
        h_1 = torch.tanh(self.proj_list[1](embed2))
        h_2=torch.tanh(self.proj_list[2](embed3))#citerseer
        gamma_0 = self.gamma[:, 0].unsqueeze(dim=-1)  # gamma_0=[3,1]
        gamma_1 = self.gamma[:, 1].unsqueeze(dim=-1)  # 形状（(rank, K+1）
        gamma_2 = self.gamma[:, 2].unsqueeze(dim=-1)  # 形状（(rank, K+1）




        eta_0 = torch.matmul(h_0, gamma_0) / self.rank  # 低秩分解[2708, 1]
        eta_1 = torch.matmul(h_1, gamma_1) / self.rank
        eta_2 = torch.matmul(h_2, gamma_2) / self.rank
        hidden = torch.matmul(embed1.unsqueeze(dim=-1), eta_0.unsqueeze(dim=-1)).squeeze(dim=-1)
        hidden = hidden + torch.matmul(embed2.unsqueeze(dim=-1), eta_1.unsqueeze(dim=-1)).squeeze(dim=-1)
        hidden = hidden + torch.matmul(embed3.unsqueeze(dim=-1), eta_2.unsqueeze(dim=-1)).squeeze(dim=-1)
        h_p_1=hidden
#############################################################################
        if self.A is None:
            self.A = adj
        seq_a = F.dropout(seq_a, self.dropout, training=self.training)

        h_a = self.MLP(seq_a)
        h_a=F.dropout(h_a, 0.2, training=self.training)
        h_p_0 = h_a
        if self.sparse:
            h_p = torch.spmm(adj, h_p_0)
            h_p = F.dropout(h_p, 0.5, training=self.training)
            h_p = torch.spmm(adj, h_p)
        else:
            h_p = torch.mm(adj, h_p_0)
            h_p = F.dropout(h_p, 0.5, training=self.training)
            h_p = torch.mm(adj, h_p)

        return h_a, h_p,h_p_1

    def embed(self, embs,seq_a , adj=None ):
        h_a = self.MLP(seq_a)
        h_a = F.dropout(h_a, 0.5, training=self.training)
        if self.sparse:
            h_p = torch.spmm(adj, h_a)
            h_p = F.dropout(h_p, 0.5, training=self.training)
            h_p = torch.spmm(adj, h_p)
        else:
            h_p = torch.mm(adj, h_a)
            h_p = F.dropout(h_p, 0.5, training=self.training)
            h_p = torch.mm(adj, h_p)
        return h_a.detach(), h_p.detach()

    def project_embeddings(self, embs):
        """
        Project embeddings of different dimensions to the same target dimension

        Args:
            embs: List of numpy arrays or tensors with potentially different dimensions

        Returns:
            List of tensors with the same embedding dimension
        """
        # Convert numpy arrays to tensors if needed
        tensor_embs = []
        for emb in embs:
            if isinstance(emb, np.ndarray):
                tensor_embs.append(torch.tensor(emb, dtype=torch.float32))
            else:
                tensor_embs.append(emb)

        # Reset linear projections
        self.linear_projections = []
        projected_embs = []

        # Get dimensions of each embedding
        dimensions = [emb.shape[1] for emb in tensor_embs]

        # Create and apply linear projections for each embedding
        for i, emb in enumerate(tensor_embs):
            # Create a linear layer to project from original dimension to target dimension
            linear = nn.Linear(dimensions[i], self.target_dim).cuda()
            self.linear_projections.append(linear)
            # Initialize the projection layer
            self.weights_init(linear)

            # Apply projection
            projected_emb = linear(emb)
            projected_emb = F.dropout(projected_emb, self.dropout, training=self.training)

            projected_embs.append(projected_emb)

        return projected_embs
