import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.copy_(torch.full_like(m.bias.data, 0.1))


class STM(torch.nn.Module):

    def __init__(self, map_emb, tgt_emb, st_hid_dim, K, dropout, emb_dim=300):
        super(STM, self).__init__()
        self.st_hid_dim = st_hid_dim
        self.K = K
        self.emb = map_emb
        self.emb_t = tgt_emb
        self.st_W = [torch.nn.Linear(emb_dim, st_hid_dim, bias=True) for i in range(K)]
        self.st_W = torch.nn.ModuleList(self.st_W)
        self.bps_W = [torch.nn.Linear(st_hid_dim, st_hid_dim, bias=False) for i in range(K)]
        self.bps_W = torch.nn.ModuleList(self.bps_W)
        for i in range(K):
            self.register_parameter('bps_b' + str(i), Parameter(torch.full((1,), 0.1)))
        self.co_W = torch.nn.Linear(K, 1)
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(dropout)
        self.init()

    def init(self):
        self.apply(weights_init)

    def spec_tensor(self, x1, x2):
        x1 = [self.dropout(self.tanh(self.st_W[i](x1))) for i in range(self.K)]
        x2 = [self.dropout(self.tanh(self.st_W[i](x2))) for i in range(self.K)]
        return x1, x2

    def bilinear_scores(self, x1, x2):
        bs = []
        for i in range(self.K):
            b1 = self.bps_W[i](x1[i])
            k = torch.einsum('bh,bh->b', (b1, x2[i]))
            k = k + eval('self.bps_b' + str(i))
            self.tanh(k)
            bs.append(k)
        bs = torch.stack(bs, dim=1)
        return bs

    def forward(self, x1, x2, tgt=False):
        if tgt:
            x1 = self.emb_t(x1)
            x2 = self.emb_t(x2)
        else:
            x1 = self.emb(x1)
            x2 = self.emb(x2)
        st1, st2 = self.spec_tensor(x1, x2)
        bs = self.bilinear_scores(st1, st2)
        bs = self.dropout(bs)
        scores = self.co_W(bs)
        return scores
