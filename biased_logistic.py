import numpy as np
from scipy import optimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot, squared_norm
from sklearn.utils.optimize import _check_optimize_result

# from sklearn.linear_model._logistic import (
#     #LabelBinarizer,
#     #_check_optimize_result,
#     #logsumexp,
#     #safe_sparse_dot,
#     #squared_norm,
# )


def loss_and_grad(x, *args):
    return _multinomial_loss_grad(x, *args)[0:2]


def _multinomial_loss_grad(w, X, Y, alpha, bias_w=0, sample_weight=None):
    n_classes = Y.shape[1]
    n_features = X.shape[1]

    if sample_weight is None:
        sample_weight = np.ones(X.shape[0])

    fit_intercept = w.size == n_classes * (n_features + 1)
    grad = np.zeros((n_classes, n_features + bool(fit_intercept)), dtype=X.dtype)
    loss, p, w = _multinomial_loss(w, X, Y, alpha, sample_weight, bias_w)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)
    grad[:, :n_features] = safe_sparse_dot(diff.T, X)
    grad[:, :n_features] += alpha * (w - bias_w)
    if fit_intercept:
        grad[:, -1] = diff.sum(axis=0)
    return loss, grad.ravel(), p


def _multinomial_loss(w, X, Y, alpha, sample_weight, bias_w):
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w - bias_w)
    p = np.exp(p, p)
    return loss, p, w


def solve_LR(
    X,
    y,
    Cs,
    sample_weight=None,
    tol=1e-4,
    max_iter=100,
    verbose=0,
    w0=None,
    bias_w=0,
    intercept=False,
):
    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    classes = np.unique(y)

    n_sample, n_feat = X.shape

    if intercept:
        X = np.concatenate([X, np.ones((n_sample, 1))], axis=1)
        if isinstance(bias_w, np.ndarray):
            bias_w = np.concatenate([bias_w, np.zeros((bias_w.shape[0], 1))], axis=1)

    if w0 is None:
        w0 = np.zeros([classes.size, X.shape[1]])
    elif intercept:
        w0 = np.concatenate([w0, np.zeros((w0.shape[0], 1))], axis=1)

    lbin = LabelBinarizer()
    Y_multi = lbin.fit_transform(y)
    if Y_multi.shape[1] == 1:
        Y_multi = np.hstack([1 - Y_multi, Y_multi])

    for i, C in enumerate(Cs):
        iprint = [-1, 50, 1, 100, 101][np.searchsorted(np.array([0, 1, 2, 3]), verbose)]
        opt_res = optimize.minimize(
            loss_and_grad,
            w0,
            method="L-BFGS-B",
            jac=True,
            args=(X, Y_multi, C, bias_w, sample_weight),
            options={"iprint": iprint, "gtol": tol, "maxiter": max_iter},
        )
        n_iter_i = _check_optimize_result("lbfgs", opt_res, max_iter, extra_warning_msg=None)
        w0, loss = opt_res.x, opt_res.fun

        n_classes = max(2, classes.size)
        multi_w0 = np.reshape(w0, (n_classes, -1))
        if n_classes == 2:
            multi_w0 = multi_w0[1][np.newaxis, :]
        coefs.append(multi_w0.copy())

    n_iter[i] = n_iter_i

    return np.array(coefs), np.array(Cs), n_iter


def create_LR_object(coef, n_class, n_iter, max_iter=100, tol=1e-4, intercept=False):
    lr = LogisticRegression(max_iter=max_iter, multi_class="multinomial", fit_intercept=intercept, tol=tol)

    if intercept:
        w0 = coef[:, :-1]
        b = coef[:, -1]
    else:
        w0 = coef
        b = 0

    class_labels = np.arange(n_class)
    lr.coef_ = w0
    lr.intercept_ = b
    lr.n_iter_ = n_iter
    lr.classes_ = class_labels
    # lr = lr.set_params(coef_=w0, intercept_=b, n_iter_=n_iter, classes_=class_labels)

    return lr
