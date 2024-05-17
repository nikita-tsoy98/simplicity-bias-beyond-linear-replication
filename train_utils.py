import random
import numpy as np

from tqdm.notebook import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from transformers import get_linear_schedule_with_warmup

def get_loss_std(z, model, loss_fn, device):
    x, y, *u = z
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    pred = model(x)
    loss = loss_fn(pred, y)
    return loss

def get_loss_one_dim(z, model, loss_fn, device):
    x, y, *u = z
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).reshape(-1, 1)
    pred = model(x)
    loss = loss_fn(pred, y)
    return loss

def train_batch(
        z, model, loss_fn, optimizer, get_loss_fn, get_loss_params, device
):
    loss = get_loss_fn(z, model, loss_fn, device, **get_loss_params)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

def train_epoch_epoch_sch(
    t,
    loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    get_loss_fn,
    get_loss_params,
    device,
    train_batch_params={},
    train_batch_fn=train_batch
):
    for z in loader:
        train_batch_fn(
            z, model, loss_fn, optimizer,
            get_loss_fn, get_loss_params, device,
            **train_batch_params)
    scheduler.step()

def train_epoch_batch_sch(
    t,
    loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    get_loss_fn,
    get_loss_params,
    device,
    train_batch_params={},
    train_batch_fn=train_batch
):
    for z in loader:
        train_batch_fn(
            z, model, loss_fn, optimizer,
            get_loss_fn, get_loss_params, device,
            **train_batch_params)
        scheduler.step()

def train_epoch_no_sch(
    t,
    loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    get_loss_fn,
    get_loss_params,
    device,
    train_batch_params={},
    train_batch_fn=train_batch
):
    for z in loader:
        train_batch_fn(
            z, model, loss_fn, optimizer,
            get_loss_fn, get_loss_params, device,
            **train_batch_params)

def logit_acc(pred, y):
    return (pred.argmax(1) == y).type(torch.float).sum()

def bin_logit_acc(pred, y):
    return (torch.sign(pred) == torch.sign(2 * y - 1)).type(torch.float).sum()

def print_acc(t, val_results):
    acc = val_results
    print(f"Epoch: {t:>3} Accuracy: {acc:2.1%}")

def print_loss(t, val_results):
    loss = val_results
    print(f"Epoch: {t:>3} Loss: {loss:.3f}")

def val_epoch(
    t, loader, model, get_loss_fn, get_loss_params, device,
    val_loss_fn=logit_acc,
    print_fn=print_acc
):
    size = len(loader.dataset)
    correct = 0

    for z in loader:
        correct += get_loss_fn(
            z, model, val_loss_fn, device, **get_loss_params).item()
    val_results = correct / size

    print_fn(t, val_results)
    return val_results

def train_model(
    epochs,
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    device,
    get_loss_params={},
    train_epoch_params={},
    val_epoch_params={},
    val_interval=1,
    get_loss_fn=get_loss_std,
    train_epoch_fn=train_epoch_batch_sch,
    val_epoch_fn=val_epoch,
    range_fn=lambda epochs: tqdm(range(epochs))
):
    model.to(device)

    for t in range_fn(epochs):
        if t % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_results = val_epoch_fn(
                    t,
                    val_loader,
                    model,
                    get_loss_fn,
                    get_loss_params,
                    device,
                    **val_epoch_params
                )

        model.train()
        train_epoch_fn(
            t,
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            get_loss_fn,
            get_loss_params,
            device,
            **train_epoch_params
        )

    model.eval()
    with torch.no_grad():
        val_results = val_epoch_fn(
            epochs,
            val_loader,
            model,
            get_loss_fn,
            get_loss_params,
            device,
            **val_epoch_params
        )
    return val_results

def prepare_data(
    root,
    data_fn,
    train_transform,
    test_transform,
    val_frac=0.25,
    train=True,
    download=False
):
    train_data = data_fn(
        root=root, train=train, download=download, transform=train_transform
    )
    val_data = data_fn(
        root=root, train=train, download=download, transform=test_transform
    )

    train_ind, val_ind = random_split(
        range(len(train_data)), [1 - val_frac, val_frac]
    )
    train_data = Subset(train_data, train_ind)
    val_data = Subset(val_data, val_ind)
    return train_data, val_data

def prepare_loaders(
    train_data,
    val_data,
    batch_size,
    num_workers,
    pin_memory=True,
    shuffle=True
):
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader

def prepare_model(model_class, num_classes, weights):
    if weights:
        model = model_class(weights=weights)
        model.fc = torch.nn.Linear(
            model.fc.in_features, num_classes
        )
    else:
        model = model_class(num_classes=num_classes)
    return model

def correct_training_params(
    epochs,
    train_loader,
    loader_params,
    optimizer_params,
    scheduler_params,
    train_params,
    lr_factor=1/2048,
    warmup_factor=1/16
):
    lr = loader_params['batch_size'] * lr_factor
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = num_training_steps * warmup_factor
    optimizer_params['lr'] = lr
    scheduler_params['num_training_steps'] = num_training_steps
    scheduler_params['num_warmup_steps'] = num_warmup_steps

def get_trained_model(
    epochs,
    data_params={},
    loader_params={},
    model_params={},
    loss_params={},
    optimizer_params={
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'nesterov': True
    },
    scheduler_params={},
    correction_params={},
    train_params={},
    data_seed=None,
    loader_seed=None,
    model_seed=None,
    train_seed=None,
    data_fn=prepare_data,
    loader_fn=prepare_loaders,
    model_fn=prepare_model,
    loss_fn=nn.CrossEntropyLoss,
    optimizer_fn=torch.optim.SGD,
    scheduler_fn=get_linear_schedule_with_warmup,
    correct_training_params_fn=correct_training_params,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    set_deterministic_seed(data_seed)
    train_data, val_data = data_fn(**data_params)

    set_deterministic_seed(loader_seed)
    train_loader, val_loader = loader_fn(
        train_data, val_data, **loader_params
    )

    set_deterministic_seed(model_seed)
    model = model_fn(**model_params)
    model.to(device)
    
    loss = loss_fn(**loss_params)

    correct_training_params_fn(
        epochs,
        train_loader,
        loader_params,
        optimizer_params,
        scheduler_params,
        train_params,
        **correction_params,
    )
    optimizer = optimizer_fn(model.parameters(), **optimizer_params)
    scheduler = scheduler_fn(optimizer, **scheduler_params)
    
    set_deterministic_seed(train_seed)
    train_model(
        epochs,
        train_loader,
        val_loader,
        model,
        loss,
        optimizer,
        scheduler,
        device,
        **train_params
    )
 
    return model
    
def reset_parameters(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()

def set_deterministic_seed(seed):
    determenistic = (seed is not None)
    torch.backends.cudnn.deterministic = determenistic
    torch.backends.cudnn.benchmark = (not determenistic)
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)