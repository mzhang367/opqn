import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class MyArgmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        return torch.argmax(input, 1)

    @staticmethod
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input


class DPQ(nn.Module):

    """
    End-to-End Supervised Product Quantization for Image Search and Retrieval (DPQ)
    https://arxiv.org/abs/1711.08589
    """
    def __init__(self, in_features, num_books, num_words):
        super(DPQ, self).__init__()
        self.in_features = in_features
        # self.out_features = out_features    # num_classes
        self.num_books = num_books
        self.num_words = num_words
        self.len_word = int(self.in_features / self.num_books)

        # self.weight = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        # mlp: project features to probability vectors.
        self.mlp = Parameter(torch.randn(self.num_books, self.len_word, self.num_words))
        self.codebook = Parameter(torch.randn(self.num_books, self.len_word, self.num_words))

        # nn.init.xavier_uniform_(self.weight)
        # nn.init.xavier_uniform_(self.mlp)
        # nn.init.xavier_uniform_(self.codebook)

    def forward(self, input):

        # ------------------------- cos(theta) & phi(theta) ---------------------------------
        x_m = torch.split(input, self.len_word, dim=1)
        xc_prod_softs = []
        xc_hard_index = []
        myargmax = MyArgmax.apply

        for i in range(self.num_books):

            xc_prod_softmax = F.softmax(x_m[i] @ self.mlp[i], dim=1)    # shape of bs * K
            xc_prod_softs.append(xc_prod_softmax)
            hard_index = myargmax(xc_prod_softmax)  # shape of (bs,) 

            xc_prod_argmax = F.one_hot(hard_index, num_classes=self.num_words)  
            xc_hard_index.append(xc_prod_argmax)

            # ----------------------------------------------
            # xc_softmax.append(xc_prod_softmax)

        xc_soft_probs = torch.stack(xc_prod_softs, dim=0)   # dim.: M * bs * K
        xc_hard_assign = torch.stack(xc_hard_index, dim=0)     # dim.: M * bs * k
        # x = torch.cat(x, dim=1)

        # ------------------------- cos(sw)---------------------------------------------------
        s_m = torch.matmul(xc_soft_probs, torch.transpose(self.codebook, 1, 2))   # construct s_m, shape of (num_books * bs * len_word)
        hard_m = torch.matmul(xc_hard_assign.float(), torch.transpose(self.codebook, 1, 2))    # shape of (num_books * bs * len_word)
        s = torch.transpose(s_m, 0, 1).reshape((input.shape[0], -1))    # bs * (MK)
        hard = torch.transpose(hard_m, 0, 1).reshape((input.shape[0], -1))
        # s_pred = F.linear(s, self.weight)
        # hard_pred = F.linear(hard, self.weight)

        return s, hard, torch.transpose(xc_soft_probs, 0, 1)



class DPQJointClassLoss(nn.Module):

    def __init__(self, num_class, feature_dim, param):

        super(DPQJointClassLoss, self).__init__()
        self.num_class = num_class
        self.feature_dim = feature_dim
        self.centers = Parameter(torch.FloatTensor(self.num_class, self.feature_dim))
        self.weight = Parameter(torch.FloatTensor(self.num_class, self.feature_dim))

        self.param = param
        nn.init.xavier_uniform_(self.centers)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, soft_x, hard_x, targets):

        soft_pred = F.linear(soft_x, self.weight)
        hard_pred = F.linear(hard_x, self.weight)

        loss_cls1 = F.cross_entropy(soft_pred, targets)
        loss_cls2 = F.cross_entropy(hard_pred, targets)
        labels_unique = torch.unique(targets).squeeze()
        loss_quant_list = []
        for label in labels_unique.tolist():
            mask = torch.nonzero(targets == label, as_tuple=False).squeeze()
            select_soft_x = torch.index_select(soft_x, 0, mask) # along dim0 to choose all the soft_x[[1,2,3],:] with the same label
            # print(select_soft_x.shape)
            select_hard_x = torch.index_select(hard_x, 0, mask)
            loss_quant = 0.5 * (select_soft_x - self.centers[label]).pow(2).sum() + \
                                0.5 * (select_hard_x - self.centers[label]).pow(2).sum()
            loss_quant_list.append(loss_quant)
        loss = loss_cls1 + loss_cls2 + self.param * sum(loss_quant_list) / soft_x.shape[0]

        return loss



