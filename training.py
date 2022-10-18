import argparse
from datetime import datetime
from collections import defaultdict

import yaml
from tqdm import tqdm
import copy
import random

import functorch
from functorch import make_functional_with_buffers, grad

from helper import Helper
from utils.utils import *

logger = logging.getLogger('logger')


def anticipate(hlpr: Helper, epoch, model, train_loader):
    attack_model = copy.deepcopy(model)
    criterion = hlpr.task.criterion
    anticipate_steps = hlpr.params.anticipate_steps

    _, attack_params, attack_buffers = make_functional_with_buffers(attack_model)
    _, weight_names, _ = functorch._src.make_functional.extract_weights(attack_model)
    _, buffer_names, _ = functorch._src.make_functional.extract_buffers(attack_model)

    optimizer = hlpr.task.make_anticipate_optimizer(attack_params + attack_buffers, epoch=epoch)
    
    for _ in range(hlpr.params.fl_attacker_local_epochs):
        for i, data in enumerate(train_loader):
            func_model, curr_params, curr_buffers = make_functional_with_buffers(model)

            batch = hlpr.task.get_batch(i, data)
            batch = hlpr.attack.synthesizer.make_backdoor_batch(batch, attack=True)

            optimizer.zero_grad()
            loss = None

            # do anticipate_steps steps
            for anticipate_i in range(anticipate_steps):
                if anticipate_i == 0:
                    # est other users' update
                    curr_params = train_with_functorch(hlpr, epoch + anticipate_i, func_model, curr_params, curr_buffers, train_loader, num_users=hlpr.params.fl_no_models-1)

                    # add attack params at step 0
                    curr_params = [(attack_params[i] + curr_params[i] * (hlpr.params.fl_no_models - 1)) / hlpr.params.fl_no_models for i in range(len(curr_params))]
                    curr_buffers = [(attack_buffers[i] + curr_buffers[i] * (hlpr.params.fl_no_models - 1)) / hlpr.params.fl_no_models for i in range(len(curr_buffers))]
                else:
                    # do normal update
                    curr_params = train_with_functorch(hlpr, epoch + anticipate_i, func_model, curr_params, curr_buffers, train_loader, num_users=hlpr.params.fl_no_models)

                # adversarial loss
                logits = func_model(curr_params, curr_buffers, batch.inputs)
                y = batch.labels

                if loss is None:
                    loss = criterion(logits, y).mean()
                else:
                    loss += criterion(logits, y).mean()

            loss.backward()

            optimizer.step()

    # copy the params back to the model
    functorch._src.make_functional.load_weights(attack_model, weight_names, attack_params)
    functorch._src.make_functional.load_buffers(attack_model, buffer_names, attack_buffers)

    return attack_model


def train_with_functorch(hlpr, epoch, func_model, params, buffers, train_loader, num_users=1):
    lr = hlpr.params.lr * hlpr.params.gamma ** (epoch)
    criterion = hlpr.task.criterion

    def compute_loss(params, buffers, x, y):
        logits = func_model(params, buffers, x)

        loss = criterion(logits, y).mean()
        return loss

    for i, data in enumerate(train_loader):
        for _ in range(hlpr.params.fl_local_epochs):
            batch = hlpr.task.get_batch(i, data)
            grads = grad(compute_loss)(params, buffers, batch.inputs, batch.labels)

            params = [p - g * lr for p, g, in zip(params, grads)]

        break

    return params


def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True, fl_attacker=False):
    criterion = hlpr.task.criterion
    model.train()

    for i, data in enumerate(train_loader):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack)
        loss.backward()
        optimizer.step()

        hlpr.report_training_losses_scales(i, epoch, fl_attacker=fl_attacker)
        if i == hlpr.params.max_batch_id:
            break

    return


def test(hlpr: Helper, epoch, model=None, backdoor=False, tb_prefix=None):
    if model is None:
        model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()
    
    test_loader = hlpr.task.test_loader

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            batch = hlpr.task.get_batch(i, data)

            if backdoor:
                # remove imgs with the target backdoor label
                batch.inputs = batch.inputs[batch.labels != hlpr.params.backdoor_label]
                batch.labels = batch.labels[batch.labels != hlpr.params.backdoor_label]
                batch.batch_size = batch.inputs.shape[0]
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)

            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)

    if tb_prefix is None:
        if backdoor:
            tb_prefix = 'Backdoor'
        else:
            tb_prefix = 'Maintask'

    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
                             tb_writer=hlpr.tb_writer,
                             tb_prefix=tb_prefix)

    return metric


def fl_run(hlpr: Helper):
    hlpr.attack.target_bias = None
    test(hlpr, -1, backdoor=False)
    test(hlpr, -1, backdoor=True)

    for epoch in range(0, hlpr.params.epochs + 1):
        run_fl_round(hlpr, epoch)
        metric = test(hlpr, epoch, backdoor=False)
        _ = test(hlpr, epoch, backdoor=True)

        hlpr.save_model(hlpr.task.model, epoch, metric)


def run_fl_round(hlpr, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = []

    num_compromised = 0
    user_update_norm = 0
    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model, epoch=epoch, hlpr=hlpr, attack=user.compromised)
        if user.compromised:
            hlpr.params.running_losses = defaultdict(list)
            hlpr.params.running_scales = defaultdict(list)

            attacker_train_loader = user.train_loader
                    
            num_compromised += 1
            fl_local_epochs = hlpr.params.fl_attacker_local_epochs
        else:
            fl_local_epochs = hlpr.params.fl_local_epochs

        for local_epoch in range(fl_local_epochs):
            if user.compromised:
                if hlpr.params.anticipate:
                    local_model = anticipate(hlpr, epoch, local_model, attacker_train_loader)
                    break
                else:
                    train(hlpr, local_epoch, local_model, optimizer,
                        attacker_train_loader, attack=True, fl_attacker=True)
            else:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=False)

        local_update = hlpr.task.get_fl_update(local_model, global_model)

        if user.compromised:
            hlpr.attack.fl_scale_update(local_update)
        else:
            update_norm = hlpr.task.get_update_norm(local_update)
            user_update_norm += update_norm

        hlpr.task.accumulate_weights(weight_accumulator, local_update)

    hlpr.task.update_global_model(weight_accumulator, global_model, hlpr, num_compromised, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--resume-model', dest='resume_model', default=None)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--random-seed', dest='random_seed', default=1)
    parser.add_argument('--optimizer', dest='optimizer', default='SGD')

    parser.add_argument('--lr', dest='lr', default=0.01)
    parser.add_argument('--epochs', dest='epochs', default=50)
    parser.add_argument('--loss-tasks', dest='loss_tasks', default=['backdoor'], nargs='+')
    parser.add_argument('--random-init', dest='random_init', default=None)
    parser.add_argument('--loss-balance', dest='loss_balance', default='MGDA')
    parser.add_argument('--fixed-scales', dest='fixed_scales', default=None, nargs='+')

    parser.add_argument('--fl-weight-scale', dest='fl_weight_scale', default=None)
    parser.add_argument('--extra-fl-weight-scale', dest='extra_fl_weight_scale', default=1)
    parser.add_argument('--fl-number-of-adversaries', dest='fl_number_of_adversaries', default=0)
    parser.add_argument('--fl-single-epoch-attack', dest='fl_single_epoch_attack', default=None, nargs='+')
    parser.add_argument('--fl-no-models', dest='fl_no_models', default=10)
    parser.add_argument('--fl-local-epochs', dest='fl_local_epochs', default=10)
    parser.add_argument('--fl-attacker-local-epochs', dest='fl_attacker_local_epochs', default=5)
    parser.add_argument('--fl-attack-freq', dest='fl_attack_freq', default=None, nargs='+')
    parser.add_argument('--fl-eta', dest='fl_eta', default=1)
    parser.add_argument('--update-method', dest='update_method', default='fedavg')
    parser.add_argument('--fl-dp-clip', dest='fl_dp_clip', default=None)
    parser.add_argument('--fl-total-participants', dest='fl_total_participants', default=100)
    parser.add_argument('--fl-sample-dirichlet', dest='fl_sample_dirichlet', action='store_true')
    parser.add_argument('--fl-dirichlet-alpha', dest='fl_dirichlet_alpha', default=1)
    parser.add_argument('--model', dest='model', action=None)
    parser.add_argument('--poisoning-proportion', dest='poisoning_proportion', default=0.5)
    parser.add_argument('--defense', dest='defense', default=None, nargs='+')

    parser.add_argument('--anticipate', dest='anticipate', action='store_true')
    parser.add_argument('--anticipate-steps', dest='anticipate_steps', default=2)

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['name'] = args.name
    params['resume_model'] = args.resume_model
    params['pretrained'] = args.pretrained
    if int(args.random_seed) >= 0:
        params['random_seed'] = int(args.random_seed)

    params['lr'] = float(args.lr)
    params['epochs'] = int(args.epochs)
    params['loss_tasks'] = args.loss_tasks
    params['loss_balance'] = args.loss_balance

    params['fl_number_of_adversaries'] = int(args.fl_number_of_adversaries)
    params['fl_no_models'] = int(args.fl_no_models)
    params['fl_local_epochs'] = int(args.fl_local_epochs)
    params['fl_attacker_local_epochs'] = int(args.fl_attacker_local_epochs)
    params['fl_eta'] = float(args.fl_eta)
    params['update_method'] = args.update_method
    params['fl_dp_clip'] = float(args.fl_dp_clip) if args.fl_dp_clip else args.fl_dp_clip
    params['fl_total_participants'] = int(args.fl_total_participants)
    params['fl_sample_dirichlet'] = args.fl_sample_dirichlet
    params['fl_dirichlet_alpha'] = float(args.fl_dirichlet_alpha)
    params['poisoning_proportion'] = float(args.poisoning_proportion)
    params['defense'] = args.defense

    params['anticipate'] = args.anticipate
    params['anticipate_steps'] = int(args.anticipate_steps)

    if args.model:
        params['model'] = args.model

    if args.optimizer:
        params['optimizer'] = args.optimizer

    if args.fl_single_epoch_attack:
        params['fl_single_epoch_attack'] = [int(i) for i in args.fl_single_epoch_attack]
        params['fl_number_of_adversaries'] = max(1, params['fl_number_of_adversaries']) # at least one 

    if args.fl_weight_scale:
        params['fl_weight_scale'] = float(args.fl_weight_scale)
    else:
        if args.update_method == 'sgd':
            params['fl_weight_scale'] = float(args.extra_fl_weight_scale) * params['fl_no_models'] / params['fl_eta']
        else:
            params['fl_weight_scale'] = float(args.extra_fl_weight_scale) * params['fl_no_models']
    
    if args.fl_attack_freq:
        start = int(args.fl_attack_freq[0])
        end = int(args.fl_attack_freq[1])
        freq = int(args.fl_attack_freq[2])

        random.seed(params['random_seed'])
        params['fl_single_epoch_attack'] = random.sample(range(start, end), freq)
        params['fl_single_epoch_attack'].sort()
        params['fl_number_of_adversaries'] = max(1, params['fl_number_of_adversaries']) # at least one 

    if args.fixed_scales:
        args.fixed_scales = [float(i) for i in args.fixed_scales]
        params['fixed_scales'] = dict(zip(args.loss_tasks, args.fixed_scales))

    helper = Helper(params)
    logger.warning(create_table(params))

    fl_run(helper)
    