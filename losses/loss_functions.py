import time

import torch

from utils.utils import record_time


def compute_all_losses_and_grads(loss_tasks, attack, model, criterion,
                                 batch, batch_back,
                                 compute_grad=None):
    grads = {}
    loss_values = {}
    for t in loss_tasks:
        # if compute_grad:
        #     model.zero_grad()
        if t == 'normal':
            loss_values[t], grads[t] = compute_normal_loss(attack.params,
                                                           model,
                                                           criterion,
                                                           batch.inputs,
                                                           batch.labels,
                                                           grads=compute_grad)
        elif t == 'backdoor':
            loss_values[t], grads[t] = compute_backdoor_loss(attack.params,
                                                             model,
                                                             criterion,
                                                             batch_back.inputs,
                                                             batch_back.labels,
                                                             grads=compute_grad)
    return loss_values, grads


def compute_normal_loss(params, model, criterion, inputs,
                        labels, grads):
    t = time.perf_counter()
    outputs = model(inputs)
    record_time(params, t, 'forward')
    loss = criterion(outputs, labels)

    if not params.dp:
        loss = loss.mean()

    if grads:
        t = time.perf_counter()
        grads = list(torch.autograd.grad(loss.mean(),
                                         [x for x in model.parameters() if
                                          x.requires_grad],
                                         retain_graph=True))
        record_time(params, t, 'backward')

    return loss, grads


def compute_backdoor_loss(params, model, criterion, inputs_back,
                          labels_back, grads=None):
    t = time.perf_counter()
    outputs = model(inputs_back)
    record_time(params, t, 'forward')
    loss = criterion(outputs, labels_back)

    if params.task == 'Pipa':
        loss[labels_back == 0] *= 0.001
        if labels_back.sum().item() == 0.0:
            loss[:] = 0.0
    if not params.dp:
        loss = loss.mean()

    if grads:
        grads = get_grads(params, model, loss)

    return loss, grads


def get_grads(params, model, loss):
    t = time.perf_counter()
    grads = list(torch.autograd.grad(loss.mean(),
                                     [x for x in model.parameters() if
                                      x.requires_grad],
                                     retain_graph=True,
                                     allow_unused=True))
    record_time(params, t, 'backward')

    grads = [i for i in grads if i is not None]

    return grads
