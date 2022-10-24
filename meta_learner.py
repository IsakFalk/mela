import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import block_diag
from torch.nn import functional as F

import util


def mean(arr):
    return sum(arr) / len(arr)


def embed_mask(n_ways, n_shots):
    ones = np.ones([n_shots, n_shots])
    arrs = [ones] * n_ways
    return block_diag(*arrs)


class MetaLS(nn.Module):
    def __init__(self, backbone, args, feat_dim, intercept=True, extra_reg=True):
        super().__init__()

        self.n_ways = args.n_ways
        self.lam = args.lam
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.n_per_class = self.n_shots + self.n_queries
        self.backbone = backbone
        self.intercept = intercept
        self.feat_dim = feat_dim
        self.extra_reg = extra_reg

        support_embed = embed_mask(self.n_ways, self.n_shots)  # *args.n_aug_support_samples
        query_embed = embed_mask(self.n_ways, args.n_queries)
        combined_embed = block_diag(support_embed, query_embed)
        embed_label = combined_embed * 2 - np.ones_like(combined_embed)
        self.embed_label = util.np_to_cuda(embed_label)

        self.logit_scale = nn.parameter.Parameter(torch.ones([]))
        self.logit_bias = nn.parameter.Parameter(torch.zeros([1, self.n_ways]))

    def add_intercept(self, xs):
        return torch.cat([xs, torch.ones([xs.shape[0], 1]).cuda()], 1)

    def proc_input(self, support_xs, support_ys, query_xs, query_ys):
        offset = support_xs.shape[0]

        combined_xs = torch.cat([support_xs, query_xs])
        combined_ys = torch.cat([support_ys, query_ys])

        combined_feat = self.backbone(combined_xs)
        return combined_feat, combined_ys, offset

    def forward(self, *batch_data):
        support_xs, support_ys, query_xs, query_ys = [x[0] for x in batch_data]

        combined_feat, combined_ys, offset = self.proc_input(support_xs, support_ys, query_xs, query_ys)

        if self.intercept:
            combined_feat = self.add_intercept(combined_feat)

        ideal_weights = self.least_square(combined_feat, combined_ys)

        support_feat = combined_feat[:offset]
        query_feat = combined_feat[offset:]

        support_weights = self.least_square(support_feat, support_ys)
        query_loss, _ = self.predict_on_task(query_feat, query_ys, support_weights)

        if self.extra_reg:
            l2_dist = torch.cdist(combined_feat, combined_feat)
            embed_loss = F.hinge_embedding_loss(l2_dist, self.embed_label, margin=1.0)

            full_loss = query_loss + embed_loss + F.mse_loss(support_weights, ideal_weights)
        else:
            full_loss = query_loss

        return full_loss

    def least_square(self, X, y, lam=None, n_ways=None):
        n = X.shape[0]
        X_t = torch.transpose(X, 0, 1)
        eye = torch.eye(n).cuda()
        if lam is None:
            lam = self.lam
        if n_ways is None:
            n_ways = self.n_ways
        A = lam * eye + torch.matmul(X, X_t)
        y_one_hot = F.one_hot(y, n_ways).float()
        alpha = torch.linalg.solve(A, y_one_hot)
        weights = torch.matmul(X_t, alpha)
        return weights

    def predict_on_task(self, xs, ys, weights):
        y_pred = torch.matmul(xs, weights)
        y_target = F.one_hot(ys, self.n_ways).float()
        loss = F.mse_loss(y_pred, y_target)

        acc = util.accuracy(y_pred, ys).item()
        return loss, acc
