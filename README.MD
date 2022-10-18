# Thinking Two Moves Ahead: Anticipating Other Users Improves Backdoor Attacks in Federated Learning

This code is the official implementation of [Thinking Two Moves Ahead](https://arxiv.org/abs/2210.09305). Our implementation is based on [Backdoor101](https://github.com/ebagdasa/backdoors101).

If you have any questions, feel free to email Yuxin (<ywen@umd.edu>).

## Dependencies

- PyTorch => 1.11.*
- torchvision >= 0.12.*
- functorch >= 0.1.1

## USAGE
non-IID CIFAR-10 baseline
```
python training.py --name baseline --fl-sample-dirichlet --fl-no-models 10 --fl-attack-freq 0 500 100 --loss-balance fixed --fixed-scales 1 --loss-tasks backdoor --update-method avg --fl-dp-clip 0.5 --fl-attacker-local-epochs 4 --fl-local-epochs 2 --lr 0.01 --params configs/cifar_fed.yaml --epochs 2000
```

non-IID CIFAR-10 anticipate
```
python training.py --name anticipate --anticipate --anticipate-steps 9 --fl-sample-dirichlet --fl-no-models 10 --fl-attack-freq 0 500 100 --loss-balance fixed --fixed-scales 1 --loss-tasks backdoor --update-method avg --fl-dp-clip 0.5 --fl-attacker-local-epochs 4 --fl-local-epochs 2 --lr 0.01 --params configs/cifar_fed.yaml --epochs 2000
```

## Results
You can view the results through running ```tensorboard --logdir runs```.
