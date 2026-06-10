from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC

from .config import get_method_config


METHOD_NAMES = (
    "zero_shot",
    "prompt_ensemble",
    "linear_probe",
    "ridge",
    "svm",
    "nearest_centroid",
    "knn",
    "soft_knn",
    "tip_adapter",
    "ape",
    "lpplusplus",
    "gda_clip",
    "proker",
)


@dataclass(frozen=True)
class MethodResult:
    method: str
    accuracy: float
    best_params: dict[str, object] = field(default_factory=dict)


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean() * 100.0)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), num_classes), dtype="float32")
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, eps)


def cosine_logits(features: np.ndarray, text_features: np.ndarray, scale: float = 100.0) -> np.ndarray:
    return scale * features.astype("float32") @ text_features.astype("float32").T


def official_tip_range(scale: float, steps: int) -> np.ndarray:
    return np.array([i * (float(scale) - 0.1) / int(steps) + 0.1 for i in range(int(steps))], dtype="float32")


def official_zero_to_scale_range(scale: float, steps: int) -> np.ndarray:
    return np.array([i * float(scale) / int(steps) for i in range(int(steps))], dtype="float32")


def _predict_sklearn(model, test_x: np.ndarray, num_classes: int) -> np.ndarray:
    if hasattr(model, "decision_function"):
        scores = model.decision_function(test_x)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        return scores.astype("float32")

    labels = model.predict(test_x)
    logits = np.full((len(labels), num_classes), -1.0, dtype="float32")
    logits[np.arange(len(labels)), labels.astype(int)] = 1.0
    return logits


def _run_classifier(method: str, train_x, train_y, test_x, num_classes: int) -> np.ndarray:
    if method == "linear_probe":
        model = LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs", n_jobs=1)
    elif method == "ridge":
        model = RidgeClassifier(alpha=1.0)
    elif method == "svm":
        model = LinearSVC(C=1.0, max_iter=10000)
    elif method == "nearest_centroid":
        model = NearestCentroid(metric="euclidean")
    elif method == "knn":
        k = min(5, len(train_y))
        model = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="uniform")
    else:
        raise KeyError(method)

    model.fit(train_x, train_y)
    return _predict_sklearn(model, test_x, num_classes)


def _soft_knn(train_x, train_y, test_x, num_classes: int, k: int = 5, temperature: float = 20.0) -> np.ndarray:
    sims = test_x @ train_x.T
    k = min(k, train_x.shape[0])
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    logits = np.zeros((test_x.shape[0], num_classes), dtype="float32")
    for row in range(test_x.shape[0]):
        row_idx = idx[row]
        weights = np.exp(temperature * sims[row, row_idx])
        weights = weights / np.maximum(weights.sum(), 1e-12)
        for neighbor, weight in zip(row_idx, weights):
            logits[row, int(train_y[neighbor])] += float(weight)
    return logits


def _search_tip_like(val_logits_text, val_x, val_y, cache_keys, cache_values, beta_grid, alpha_grid):
    best = (-1.0, None, None)
    affinity = val_x @ cache_keys.T
    for beta in beta_grid:
        cache_affinity = np.exp(-float(beta) * (1.0 - affinity))
        cache_logits_base = cache_affinity @ cache_values
        for alpha in alpha_grid:
            logits = val_logits_text + float(alpha) * cache_logits_base
            acc = accuracy_from_logits(logits, val_y)
            if acc > best[0]:
                best = (acc, float(alpha), float(beta))
    return best


def _tip_adapter(train_x, train_y, val_x, val_y, test_x, text_features, num_classes, cfg):
    if cfg.get("augmented_train_x") is not None:
        aug_x = cfg["augmented_train_x"]
        augment_epoch = int(cfg.get("augment_epoch", 1))
        train_x = l2_normalize(aug_x.reshape(augment_epoch, train_x.shape[0], train_x.shape[1]).mean(axis=0)).astype("float32")
    cache_values = one_hot(train_y, num_classes)
    val_text = cosine_logits(val_x, text_features)
    test_text = cosine_logits(test_x, text_features)
    beta_grid = official_tip_range(cfg.get("search_scale", [20, 20])[0], cfg.get("search_step", [200, 20])[0])
    alpha_grid = official_tip_range(cfg.get("search_scale", [20, 20])[1], cfg.get("search_step", [200, 20])[1])
    _, alpha, beta = _search_tip_like(val_text, val_x, val_y, train_x, cache_values, beta_grid, alpha_grid)
    test_cache = np.exp(-beta * (1.0 - test_x @ train_x.T)) @ cache_values
    return test_text + alpha * test_cache, {"alpha": alpha, "beta": beta, "search_scale": cfg.get("search_scale"), "search_step": cfg.get("search_step")}


def _ape(train_x, train_y, val_x, val_y, test_x, text_features, num_classes, cfg):
    if cfg.get("augmented_train_x") is not None:
        aug_x = cfg["augmented_train_x"]
        augment_epoch = int(cfg.get("augment_epoch", 1))
        train_x = l2_normalize(aug_x.reshape(augment_epoch, train_x.shape[0], train_x.shape[1]).mean(axis=0)).astype("float32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = torch.tensor(train_x, dtype=torch.float32, device=device)
    val = torch.tensor(val_x, dtype=torch.float32, device=device)
    test = torch.tensor(test_x, dtype=torch.float32, device=device)
    labels = torch.tensor(train_y, dtype=torch.long, device=device)
    text = torch.tensor(text_features, dtype=torch.float32, device=device)
    cache_values = F.one_hot(labels, num_classes=num_classes).float()

    grouped = []
    for label in range(num_classes):
        grouped.append(train[labels == label])
    min_shots = min(x.shape[0] for x in grouped)
    class_feats = torch.stack([x[:min_shots] for x in grouped], dim=0)
    feats = torch.cat([text.unsqueeze(1), class_feats], dim=1)
    class_means = feats.mean(dim=1)
    sim_sum = torch.zeros(text.shape[1], device=device)
    count = 0
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                sim_sum += class_means[i] * class_means[j]
                count += 1
    sim = sim_sum / max(count, 1)
    w = cfg.get("w_training_free", [0.5, 0.5])
    criterion = -float(w[0]) * sim + float(w[1]) * torch.var(text.T, dim=1)
    feat_num = min(int(cfg.get("training_free_feat_num", 1024)), text.shape[1])
    selected = torch.topk(criterion, k=feat_num).indices

    train_s = F.normalize(train[:, selected], dim=-1)
    val_s = F.normalize(val[:, selected], dim=-1)
    test_s = F.normalize(test[:, selected], dim=-1)
    text_s = F.normalize(text[:, selected], dim=-1)

    key_logits = F.softmax(train_s @ text_s.T, dim=1)
    cache_div = torch.sum(cache_values * torch.log2((cache_values + 1e-6) / (key_logits + 1e-6)), dim=1, keepdim=True)
    val_text = 100.0 * val @ text.T
    test_text = 100.0 * test @ text.T

    search_scale = cfg.get("search_scale", [20, 20, 1])
    search_step = cfg.get("search_step", [200, 20, 20])
    beta_grid = torch.tensor(official_tip_range(search_scale[0], search_step[0]), device=device)
    alpha_grid = torch.tensor(official_tip_range(search_scale[1], search_step[1]), device=device)
    gamma_grid = torch.tensor(official_zero_to_scale_range(search_scale[2], search_step[2]), device=device)
    best = (-1.0, None, None, None)
    val_affinity = val_s @ train_s.T
    val_labels = torch.tensor(val_y, dtype=torch.long, device=device)
    for beta in beta_grid:
        val_base = torch.exp(-beta * (1.0 - val_affinity))
        for alpha in alpha_grid:
            for gamma in gamma_grid:
                soft_cache_values = cache_values * torch.exp(cache_div * gamma)
                cache_logits = val_base @ soft_cache_values
                logits = val_text + alpha * cache_logits
                acc = (logits.argmax(dim=1) == val_labels).float().mean().item() * 100.0
                if acc > best[0]:
                    best = (acc, float(alpha), float(beta), float(gamma))

    _, alpha, beta, gamma = best
    test_cache_values = cache_values * torch.exp(cache_div * gamma)
    test_cache = torch.exp(-beta * (1.0 - test_s @ train_s.T)) @ test_cache_values
    logits = test_text + alpha * test_cache
    return logits.cpu().numpy(), {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "feat_num": feat_num,
        "search_scale": search_scale,
        "search_step": search_step,
        "w_training_free": w,
    }


def _lpplusplus(train_x, train_y, val_x, val_y, test_x, text_features, num_classes, cfg):
    if cfg.get("augmented_train_x") is not None and cfg.get("augmented_train_y") is not None:
        train_x = l2_normalize(cfg["augmented_train_x"]).astype("float32")
        train_y = cfg["augmented_train_y"]

    epochs = int(cfg.get("train_epoch", 300))
    shots = int(cfg.get("shots", max(1, train_x.shape[0] // max(num_classes, 1))))
    seed = int(cfg.get("seed", 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x = torch.tensor(train_x, dtype=dtype, device=device)
    y = torch.tensor(train_y, dtype=torch.long, device=device)
    vx = torch.tensor(val_x, dtype=dtype, device=device)
    vy = torch.tensor(val_y, dtype=torch.long, device=device)
    tx = torch.tensor(test_x, dtype=dtype, device=device)
    text = torch.tensor(text_features, dtype=dtype, device=device)

    centroids = torch.stack([x[y == c].sum(dim=0) for c in range(num_classes)], dim=0)
    classifier = torch.nn.Linear(x.shape[1], num_classes, bias=True).to(device)
    classifier.weight.data.copy_(centroids)

    ff_t = torch.linalg.eigvalsh(x.T @ x).max().item()
    lr_w = float((4 * x.shape[0]) / max(ff_t, 1e-6))
    ft_t = x @ text.T
    lr_alpha = float(x.shape[0] / max((ft_t.pow(2).sum(dim=0).max().item() * 4), 1e-6))
    alpha_tilde = torch.stack([ft_t[y == c, c].mean() for c in range(num_classes)])
    alpha_init = (250.0 / max(shots, 1)) * (alpha_tilde.double() * shots)
    alpha_vec = torch.autograd.Variable(
        torch.ones(1, num_classes, dtype=dtype, device=device) * alpha_init.mean().float(),
        requires_grad=True,
    )

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr_w, momentum=0.9)

    best = (-1.0, None)
    best_epoch = 0
    update_interval = int(cfg.get("alpha_update_interval", 10))
    for epoch in range(epochs):
        logits = classifier(x) + torch.ones(x.shape[0], 1, dtype=dtype, device=device) @ alpha_vec * (x @ text.T)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update_interval > 0 and (epoch + 1) % update_interval == 0 and alpha_vec.grad is not None:
            alpha_vec.data -= lr_alpha * alpha_vec.grad.data

        with torch.no_grad():
            val_logits = classifier(vx) + torch.ones(vx.shape[0], 1, dtype=dtype, device=device) @ alpha_vec * (vx @ text.T)
            acc = (val_logits.argmax(dim=1) == vy).float().mean().item() * 100.0
            if acc >= best[0]:
                test_logits = classifier(tx) + torch.ones(tx.shape[0], 1, dtype=dtype, device=device) @ alpha_vec * (tx @ text.T)
                best = (acc, test_logits.detach().cpu().numpy())
                best_epoch = epoch

    return best[1], {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "lr_w": lr_w,
        "lr_alpha": lr_alpha,
        "alpha_init": float(alpha_init.mean().item()),
        "alpha_update_interval": update_interval,
        "classifier_centroid": cfg.get("classifier_centroid", "class_sum"),
    }


def _gda_clip(train_x, train_y, val_x, val_y, test_x, text_features, num_classes, cfg):
    if cfg.get("augmented_train_x") is not None and cfg.get("augmented_train_y") is not None:
        train_x = l2_normalize(cfg["augmented_train_x"]).astype("float32")
        train_y = cfg["augmented_train_y"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(train_x, dtype=torch.float32, device=device)
    y = torch.tensor(train_y, dtype=torch.long, device=device)
    val = torch.tensor(val_x, dtype=torch.float32, device=device)
    test = torch.tensor(test_x, dtype=torch.float32, device=device)
    text = torch.tensor(text_features, dtype=torch.float32, device=device)

    mus = torch.stack([x[y == c].mean(dim=0) for c in range(num_classes)], dim=0)
    centered = torch.cat([x[y == c] - mus[c] for c in range(num_classes)], dim=0)
    cov = centered.T.cov()
    cov_inv = x.shape[1] * torch.linalg.pinv((centered.shape[0] - 1) * cov + cov.trace() * torch.eye(x.shape[1], device=device))
    prior = torch.ones(num_classes, device=device) / num_classes
    w = mus @ cov_inv
    b = prior.log() - torch.einsum("cd,dk,ck->c", mus, cov_inv, mus) / 2.0

    val_text = 100.0 * val @ text.T
    test_text = 100.0 * test @ text.T
    val_gda = val @ w.T + b
    test_gda = test @ w.T + b
    best = (-1.0, 0.1)
    alpha_grid = cfg.get("alpha_grid", [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    for alpha in alpha_grid:
        logits = val_text + alpha * val_gda
        acc = (logits.argmax(dim=1).cpu().numpy() == val_y).mean() * 100.0
        if acc > best[0]:
            best = (float(acc), alpha)
    logits = test_text + best[1] * test_gda
    return logits.cpu().numpy(), {"alpha": best[1], "alpha_grid": alpha_grid}


def _proker(train_x, train_y, val_x, val_y, test_x, text_features, num_classes, cfg):
    if cfg.get("augmented_train_x") is not None and cfg.get("augmented_train_y") is not None:
        train_x = l2_normalize(cfg["augmented_train_x"]).astype("float32")
        train_y = cfg["augmented_train_y"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(train_x, dtype=torch.float32, device=device)
    y = torch.tensor(train_y, dtype=torch.long, device=device)
    val = torch.tensor(val_x, dtype=torch.float32, device=device)
    test = torch.tensor(test_x, dtype=torch.float32, device=device)
    text = torch.tensor(text_features, dtype=torch.float32, device=device)
    cache = F.one_hot(y, num_classes=num_classes).float()
    text_shots = x @ text.T
    text_val = val @ text.T
    text_test = test @ text.T

    def rbf(a, b, beta):
        return torch.exp(-beta * (1.0 - a @ b.T))

    best = (-1.0, None, None)
    beta_cfg = cfg.get("beta", [0.1, 20.0, 10])
    lmbda_cfg = cfg.get("lmbda", [0.001, 1.0, 10])
    for beta in torch.linspace(float(beta_cfg[0]), float(beta_cfg[1]), int(beta_cfg[2]), device=device):
        k_ss = rbf(x, x, beta)
        k_vs = rbf(val, x, beta)
        eye = torch.eye(x.shape[0], device=device)
        for lmbda in torch.linspace(float(lmbda_cfg[0]), float(lmbda_cfg[1]), int(lmbda_cfg[2]), device=device):
            alpha_i = torch.linalg.solve((1.0 / lmbda) * k_ss + eye, cache - text_shots)
            logits = text_val + k_vs @ alpha_i
            acc = (logits.argmax(dim=1).cpu().numpy() == val_y).mean() * 100.0
            if acc > best[0]:
                best = (float(acc), float(beta), float(lmbda))

    _, beta, lmbda = best
    k_ss = rbf(x, x, beta)
    alpha_i = torch.linalg.solve((1.0 / lmbda) * k_ss + torch.eye(x.shape[0], device=device), cache - text_shots)
    logits = text_test + rbf(test, x, beta) @ alpha_i
    return logits.cpu().numpy(), {"beta": beta, "lambda": lmbda, "beta_grid": beta_cfg, "lambda_grid": lmbda_cfg}


def run_feature_method(
    method: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    text_features: np.ndarray,
    dataset: str | None = None,
    method_config: dict | None = None,
    augmented_train_x: np.ndarray | None = None,
    augmented_train_y: np.ndarray | None = None,
) -> MethodResult:
    method = method.lower()
    num_classes = text_features.shape[0]
    cfg = get_method_config(method, dataset)
    if method_config:
        cfg.update(method_config)
    if augmented_train_x is not None:
        cfg["augmented_train_x"] = augmented_train_x
    if augmented_train_y is not None:
        cfg["augmented_train_y"] = augmented_train_y

    if method in {"zero_shot", "prompt_ensemble"}:
        logits = cosine_logits(test_x, text_features)
        params = {}
    elif method in {"linear_probe", "ridge", "svm", "nearest_centroid", "knn"}:
        logits = _run_classifier(method, train_x, train_y, test_x, num_classes)
        params = {}
    elif method == "soft_knn":
        logits = _soft_knn(train_x, train_y, test_x, num_classes)
        params = {"k": min(5, len(train_y)), "temperature": 20.0}
    elif method == "tip_adapter":
        logits, params = _tip_adapter(train_x, train_y, val_x, val_y, test_x, text_features, num_classes, cfg)
    elif method == "ape":
        logits, params = _ape(train_x, train_y, val_x, val_y, test_x, text_features, num_classes, cfg)
    elif method == "lpplusplus":
        logits, params = _lpplusplus(train_x, train_y, val_x, val_y, test_x, text_features, num_classes, cfg)
    elif method == "gda_clip":
        logits, params = _gda_clip(train_x, train_y, val_x, val_y, test_x, text_features, num_classes, cfg)
    elif method == "proker":
        logits, params = _proker(train_x, train_y, val_x, val_y, test_x, text_features, num_classes, cfg)
    else:
        raise KeyError(f"Unknown method {method!r}. Available: {METHOD_NAMES}")

    return MethodResult(method=method, accuracy=accuracy_from_logits(logits, test_y), best_params=params)
