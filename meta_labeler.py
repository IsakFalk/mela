import numpy as np
import torch
import torch.nn as nn

import util


class MetaLabeler(nn.Module):
    def __init__(self, backbone, args, feat_dim):
        super(MetaLabeler, self).__init__()

        # We can only have
        assert isinstance(args.n_ways, int) and isinstance(args.n_shots, int)
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.n_per_class = self.n_shots + self.n_queries
        self.backbone = backbone
        self.feat_dim = feat_dim

        # parameters for labeling
        self.K = args.K  # Number of starting centroids
        self.centroid_counter = np.zeros([self.K])
        self.filtering_counter = np.zeros([self.K])

        self.register_buffer("centroid", torch.zeros(self.K, self.feat_dim))
        self.ptr = 0

    def group_by_class(self, combined_feat, offset):
        support_feat = combined_feat[:offset]
        query_feat = combined_feat[offset:]
        support_feat = support_feat.view(self.n_ways, -1, self.feat_dim)
        query_feat = query_feat.view(self.n_ways, -1, self.feat_dim)
        return torch.cat([support_feat, query_feat], 1)

    @torch.no_grad()
    def populate_queue(self, keys, idx=None):
        if idx is None:
            batch_size = keys.shape[0]

            assert self.K % batch_size == 0  # for simplicity
            self.centroid[self.ptr : self.ptr + batch_size] = keys
            self.ptr = (self.ptr + batch_size) % self.K
        else:
            self.centroid[idx] = keys

    @torch.no_grad()
    def cluster_task(self, xs, real_ys):
        # Assign current labels to xs according
        # distance to centroids while grouping instances
        # according to local labels
        # NOTE: real_ys are not used, so we are not cheating
        class_keys, class_labels = self.pseudo_labels(xs)
        if class_labels is not None:
            keys = self.centroid[class_labels]

            np_idx = util.cuda_to_np(class_labels).astype(int)
            before = self.centroid_counter[np_idx]
            self.filtering_counter[np_idx] += 1
            after = before + 1
            self.centroid_counter[np_idx] = after
            ratio = before / after
            keys = keys * util.np_to_cuda(ratio).view(-1, 1) + class_keys / util.np_to_cuda(after).view(-1, 1)

            self.populate_queue(keys, idx=class_labels)
            return True
        return False

    @torch.no_grad()
    def init_centroid(self, xs, ys):
        combined_feat = self.backbone(xs)
        feat_by_class = combined_feat.reshape(self.n_ways, -1, self.feat_dim)
        feat_by_class = feat_by_class.mean(1)
        self.populate_queue(feat_by_class)

    @torch.no_grad()
    def remove_cluster(self, n_ways, std_factor):
        counter = self.filtering_counter
        p = n_ways / counter.shape[0]
        mean = np.mean(counter)
        std = np.sqrt(p * (1 - p) * np.sum(counter) / n_ways)
        threshold = np.max([mean - std_factor * std, 1])

        idx = counter > threshold

        self.centroid = self.centroid[idx]
        self.centroid_counter = self.centroid_counter[idx]
        self.K = self.centroid.shape[0]
        self.filtering_counter = np.zeros(self.K)

    def load_cluster(self, file):
        self.centroid = torch.load(file)
        self.K = self.centroid.shape[0]

    @torch.no_grad()
    def pseudo_labels(self, combined_xs, topk=1):
        combined_feat = self.backbone(combined_xs)

        # Create kernel mean embedding (KME) from the instances
        # according to local labels
        feat_by_class = combined_feat.reshape(self.n_ways, -1, self.feat_dim)
        class_keys = feat_by_class.mean(1)

        # Get distance from KMEs to centroids
        class_vote = torch.cdist(class_keys, self.centroid)
        vals, idx = torch.topk(class_vote, topk, largest=False)

        top_choice = idx[:, 0].cuda()

        if topk == 1:
            fake_labels = idx[:, 0].cuda()
        else:
            fake_labels = torch.zeros(self.n_ways, self.K).cuda()
            non_zeros = 1 / vals
            norm_prob = non_zeros / torch.sum(non_zeros, dim=1, keepdim=True)

            fake_labels.scatter_(1, idx, norm_prob)

        # We return None, None (i.e. failure) if any of the local labels
        # are mapped to the same centroid
        if torch.unique(top_choice).shape[0] == self.n_ways:
            return class_keys, fake_labels
        return None, None

    def label_samples(self, combined_xs, topk=1):
        _, class_labels = self.pseudo_labels(combined_xs, topk=topk)
        if class_labels is not None:
            return (
                torch.repeat_interleave(class_labels, self.n_per_class, dim=0),
                class_labels,
            )
        return None, None
