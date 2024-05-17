import torch
from torch import nn
from torch.utils.data import TensorDataset

from torchvision.models.feature_extraction import create_feature_extractor

import numpy as np

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

import warnings

from train_utils import *

def gen_sym_data(n, d, mu, delta, sigma):
    x = torch.empty((n // 16, d), dtype=torch.float64).normal_(0, sigma)

    small = (x**2).sum(1) < delta**2
    x = x[small]

    shift = torch.zeros_like(x[0])
    shift[0] = mu
    x += shift

    refl = torch.ones_like(x[0])
    refl[0] = -1
    x = torch.vstack((x, refl * x))
    refl = torch.ones_like(x[0])
    refl[1] = -1
    x = torch.vstack((x, refl * x))
    refl = torch.ones_like(x[0])
    refl[2:] = -1
    x = torch.vstack((x, refl * x))

    rotation = torch.eye(d, dtype=x.dtype)
    rotation[0, 0] = 0
    rotation[0, 1] = 1
    rotation[1, 0] = 1
    rotation[1, 1] = 0
    x = torch.vstack((x, x @ rotation))
    y = torch.zeros(len(x), dtype=x.dtype)
    y[len(x)//2:] = 1

    return TensorDataset(x, y)

def gen_skew_data(n, d, mu, delta, sigma, alpha):
    x = torch.empty((n // 4, d), dtype=torch.float64).normal_(0, sigma)

    small = (x[:, :2]**2).sum(1) < delta**2
    x = x[small]

    shift = torch.zeros_like(x[0])
    shift[0] = mu
    x += shift
    
    refl = torch.ones_like(x[0])
    refl[0] = -1
    x = torch.vstack((x, refl * x))

    rotation = torch.eye(d, dtype=x.dtype)
    rotation[0, 0] = np.cos(alpha)
    rotation[0, 1] = np.sin(alpha)
    rotation[1, 0] = np.sin(alpha)
    rotation[1, 1] = -np.cos(alpha)
    x = torch.vstack((x, x @ rotation))
    y = torch.zeros(len(x), dtype=x.dtype)
    y[len(x)//2:] = 1

    return TensorDataset(x, y)

def gen_same_data(n, data_fn, data_params):
    train_data = data_fn(n, **data_params)
    val_data = train_data

    return train_data, val_data

def gen_random_model(d, m, scale):
    model = nn.Sequential(
        nn.Linear(d, m, bias=False),
        nn.ReLU(),
        nn.Linear(m, 1, bias=False)
    )
    model.double()
    with torch.no_grad():
        for module in model:
            if hasattr(module, "weight"):
                module.weight.copy_(module.weight * scale)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.copy_(module.bias * scale)
    return model

def gen_2d_model(init_weight):
    model = nn.Sequential(
        nn.Linear(2, 4, bias=False),
        nn.ReLU(),
        nn.Linear(4, 1, bias=False)
    )
    model.double()
    with torch.no_grad():
        model[0].weight.copy_(init_weight)
        model[2].weight.copy_(
            torch.tensor([[-1, 1, -1, 1.0]]) *\
            (model[0].weight**2).sum(1).sqrt())
    return model

def log_epoch(
    t, loader, model, get_loss_fn, get_loss_params, device,
    weights_path 
):
    weights_path[t] = model[0].weight.cpu().numpy()
    return None


def my_correct_training_params(
    epochs,
    train_loader,
    loader_params,
    optimizer_params,
    scheduler_params,
    train_params,
    lr=2**(-7)
):
    optimizer_params['lr'] = lr

def gen_train_data(n, d, mu, delta, sigma):
    x = torch.empty((n, d), dtype=torch.float64).normal_(0, sigma)

    small = (x**2).sum(1) < delta**2
    x = x[small]
    x = x[:len(x)-(len(x) % 4)]

    x[:len(x)//4, 0] += mu
    x[len(x)//4:len(x)//2, 0] -= mu
    x[len(x)//2:3*len(x)//4, 1] += mu
    x[3*len(x)//4:, 1] -= mu

    y = torch.zeros(len(x), dtype=x.dtype)
    y[len(x)//2:] = 1
    
    return TensorDataset(x, y)

def gen_ood_data(n, d, mu, delta, sigma):
    x = torch.empty((n//2, d), dtype=torch.float64).normal_(0, sigma)

    small = (x[:, :2]**2).sum(1) < delta**2
    x = x[small]

    s = len(x) // 6
    x[:s, 0] += mu
    x[s:2*s, 0] -= mu
    x[2*s:3*s, 0] += mu / 3
    x[2*s:3*s, 1] += 3**0.5 * mu / 3

    x[3*s:4*s, 1] += mu
    x[4*s:5*s, 1] -= mu
    x[5*s:, 0] += 3**0.5 * mu / 3
    x[5*s:, 1] += mu / 3

    y = torch.zeros(len(x), dtype=x.dtype)
    y[len(x)//2:] = 1

    refl = torch.ones_like(x[0])
    refl[2:] = -1
    x = torch.vstack((x, refl * x))
    y = torch.hstack((y, y))

    return TensorDataset(x, y)

def gen_data(
    n, data_fn, data_params, val_frac=0.25
):
    val_size = int(n * val_frac)
    train_data = data_fn(n - val_size, **data_params)
    val_data = data_fn(val_size, **data_params)
    return train_data, val_data

def gen_small_model(d, m1, m2, scale):
    model = nn.Sequential(
        nn.Linear(d, m1),
        nn.ReLU(),
        nn.Linear(m1, m2),
        nn.ReLU(),
        nn.Linear(m2, 1)
    )
    model.double()
    with torch.no_grad():
        for i in range(0, 5, 2):
            model[i].weight.copy_(model[i].weight * scale)
            model[i].bias.copy_(model[i].bias * scale)
    return model

def extract_features(
    extractor, loader, device, epochs=1
):
    extractor.eval()
    size = epochs * len(loader.dataset)
    x, *z = next(iter(loader))
    feat = extractor(x.to(device))['features'].flatten(1)
    feat_full = torch.empty((size, *feat.shape[1:]), dtype=feat.dtype)
    z_full = tuple(torch.empty((size, *y.shape[1:]), dtype=y.dtype) for y in z)
    pos = 0

    with torch.no_grad():
        for _ in range(epochs):
            for x, *z in loader:
                x = x.to(device, non_blocking=True)
                feat = extractor(x)['features'].flatten(1).cpu()
                feat_full[pos:pos+len(x)] = feat
                for y, y_full in zip(z, z_full):
                    y_full[pos:pos+len(x)] = y
                pos += len(x)

    return feat_full.numpy(), *(y_full.numpy() for y_full in z_full)

def ood_val(
    model,
    feature_index,
    ood_val_data,
    ood_test_data,
    loader_params,
    warm_start_restarts,
    loader_fn,
    ood_reg_fn,
    device,
):
    ood_val_loader, ood_test_loader = loader_fn(
        ood_val_data, ood_test_data,
        **loader_params, pin_memory=False, shuffle=False)
    extractor = create_feature_extractor(model, {feature_index: 'features'})
    val_feat, val_y, *z = extract_features(extractor, ood_val_loader, device)
    test_feat, test_y, *z = extract_features(extractor, ood_test_loader, device)
    val_scale = ((val_feat**2).sum() / val_feat.shape[0])**0.5
    val_feat = val_feat / val_scale
    test_feat = test_feat / val_scale
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for _ in range(warm_start_restarts-1):
            ood_reg_fn.fit(val_feat, val_y)
    ood_acc = ood_reg_fn.fit(val_feat, val_y).score(test_feat, test_y)
    reg_scale = (ood_reg_fn.coef_**2).sum() / ood_reg_fn.coef_.size
    return ood_acc, reg_scale

def get_scale(model):
    scale = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            scale += (parameter**2).sum()
    return scale.item()

def ood_epoch( 
    t, loader, model, get_loss_fn, get_loss_params, device,
    stats,
    feature_index, ood_val_data, ood_test_data, loader_params,
    warm_start_restarts = 1,
    loader_fn=prepare_loaders,
    ood_reg_fn=LogisticRegression(),
    val_loss_fn=logit_acc,
    print_fn=print_acc,
    print_ood=lambda x, y: print(f"OOD Acc: {x:2.1%} Scale: {y:.3f}")
):
    val_acc = val_epoch(
        t, loader, model, get_loss_fn, get_loss_params, device,
        val_loss_fn, print_fn
    )
    ood_acc, reg_scale = ood_val(
        model,
        feature_index,
        ood_val_data,
        ood_test_data, 
        loader_params,
        warm_start_restarts,
        loader_fn,
        ood_reg_fn,
        device)
    scale = get_scale(model)
    print_ood(ood_acc, scale)
    stats[0].append(val_acc)
    stats[1].append(ood_acc)
    stats[2].append(scale**0.5)
    stats[3].append(reg_scale**0.5)
    return val_acc