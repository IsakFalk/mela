from collections import defaultdict

import numpy as np
import scipy
import torch
from scipy.stats import t
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import Dataset
from tqdm import tqdm

import util


class FeatStore(Dataset):
    def __init__(self, xs, ys, args, db_size, n_ways, n_shots, fix_seed=True, is_norm=False):
        super().__init__()
        self.data = xs
        self.labels = ys
        self.labels = self.labels - np.min(self.labels)

        if is_norm:
            self.data = np_normalize(xs)

        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = args.n_queries
        self.n_per_class = self.n_shots + self.n_queries
        self.db_size = db_size
        self.local_cls = np.arange(self.n_ways)
        self.fix_seed = fix_seed

        self.label_to_index = defaultdict(list)
        for i in range(self.data.shape[0]):
            self.label_to_index[self.labels[i]].append(i)
        self.classes = list(self.label_to_index.keys())

        for key in self.label_to_index:
            self.label_to_index[key] = np.asarray(self.label_to_index[key]).astype(int)

    def _random_item(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)

        xs = []
        task_class_to_idx = []
        for cls in cls_sampled:
            try:
                idx_sampled = np.random.choice(self.label_to_index[cls], self.n_per_class, False)
            except:
                # Fungi has some classes with only a few samples
                idx_sampled = np.random.choice(self.label_to_index[cls], self.n_per_class, True)
            task_class_to_idx.append(idx_sampled)
            xs.append(self.data[idx_sampled])
        xs = np.array(xs)
        task_class_to_idx = np.array(task_class_to_idx)

        return xs, cls_sampled, task_class_to_idx

    def __getitem__(self, item):
        xs, _, _ = self._random_item(item)

        data_dims = xs.shape[2:]

        support_xs, query_xs = np.split(xs, [self.n_shots], axis=1)

        support_xs = support_xs.reshape((-1, *data_dims))
        query_xs = query_xs.reshape((-1, *data_dims))

        return (
            support_xs,
            np.repeat(self.local_cls, self.n_shots),
            query_xs,
            np.repeat(self.local_cls, self.n_queries),
        )

    def __len__(self):
        return self.db_size


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1 + confidence) / 2.0, n - 1)
    return m, h


def np_normalize(x):
    eps = 1e-8
    norm = np.sqrt(np.sum(np.power(x, 2), axis=1, keepdims=True))
    return x / (norm + eps)


def normalize(x):
    eps = 1e-8
    norm = x.pow(2).sum(1, keepdim=True).pow(1.0 / 2)
    out = x.div(norm + eps)
    return out


def np_check_nan_or_inf(x):
    return np.isnan(x).any() or np.isinf(x).any()


def torch_check_nan_or_inf(x):
    return torch.isnan(x).any() or torch.isinf(x).any()


def meta_test_new(net, testloader, val_n_shots, logger, opt, classifier="LR", feats=None):
    if net is not None:
        net = net.eval()
    acc = []

    if hasattr(net, "backbone"):
        encoder = net.backbone
    else:
        encoder = net

    if feats is None:
        feats = []
        labels = []
        with torch.no_grad():
            for i, (xs, ys, idx) in enumerate(tqdm(testloader)):
                nan_xs = torch_check_nan_or_inf(xs)
                if nan_xs:
                    logger.info(f"Corrupt xs found in batch {i}, skipping")
                    continue
                feat = encoder(xs.cuda())
                # Weird nans / infs on cuda, but
                # seem to be fixed by just trying again
                nan_feat = torch_check_nan_or_inf(feat)
                counter = 0
                if nan_feat:
                    logger.info(f"Corrupt feat found in batch {i}")
                while nan_feat and counter <= 5:
                    counter += 1
                    feat = encoder(xs.cuda())
                    nan_feat = torch_check_nan_or_inf(feat)
                if counter > 5:
                    logger.info(f"Feats still corrupted after retrying {counter} times, skipping batch")
                    util.save_routine(
                        -1, net, {}, f"{opt.model_path}/{opt.model_name}_xvalnshots{val_n_shots}_best_CORRUPTED"
                    )
                    continue
                elif counter > 0:
                    logger.info(f"Retrying batch succeeded on try {counter}!")
                feats.append(util.cuda_to_np(feat))
                labels.append(ys)
        feats = np.concatenate(feats, axis=0)
        labels = np.concatenate(labels, axis=0)
    else:
        labels = testloader.dataset.labels
    meta_dataset = FeatStore(
        feats, labels, opt, opt.test_db_size, opt.n_ways, val_n_shots, fix_seed=True, is_norm=opt.is_norm
    )

    for i in range(opt.test_db_size):
        acc.append(base_learner(meta_dataset[i], opt, classifier=classifier))

    return acc


def base_learner(data, opt, classifier="LR"):
    support_features, support_ys, query_features, query_ys = data[:4]

    if classifier == "LR":
        clf = LogisticRegression(
            penalty="l2",
            C=opt.test_C,
            random_state=0,
            solver="lbfgs",
            max_iter=1000,
            fit_intercept=opt.use_bias,
            multi_class="multinomial",
        )
        clf.fit(support_features, support_ys)
        query_ys_pred = clf.predict(query_features)
    elif classifier == "SVM":
        clf = make_pipeline(
            StandardScaler(),
            SVC(gamma="auto", C=1, kernel="linear", decision_function_shape="ovr"),
        )
        clf.fit(support_features, support_ys)
        query_ys_pred = clf.predict(query_features)
    elif classifier == "NN":
        query_ys_pred = NN(support_features, support_ys, query_features)
    elif classifier == "Cosine":
        query_ys_pred = Cosine(support_features, support_ys, query_features)
    elif classifier == "Proto":
        query_ys_pred = Proto(support_features, support_ys, query_features, opt)
    else:
        raise NotImplementedError("classifier not supported: {}".format(classifier))

    return metrics.accuracy_score(query_ys, query_ys_pred)


def Proto(support, support_ys, query, opt):
    """Protonet classifier"""
    nc = support.shape[-1]
    support = np.reshape(support, (-1, 1, opt.n_ways, opt.n_shots, nc))
    support = support.mean(axis=3)
    batch_size = support.shape[0]
    query = np.reshape(query, (batch_size, -1, 1, nc))
    logits = -((query - support) ** 2).sum(-1)
    pred = np.argmax(logits, axis=-1)
    pred = np.reshape(pred, (-1,))
    return pred


def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred
