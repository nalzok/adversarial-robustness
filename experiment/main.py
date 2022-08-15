import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache.compilation_cache import initialize_cache
from flax.jax_utils import replicate, unreplicate
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import click

from .train import create_train_state, create_finetune_state, train_step, update_batch_stats_step, cross_replica_mean, test_step
from .attacks import pgd_untargeted


@click.command()
@click.option('--adv_epsilon', type=float, required=True)
@click.option('--pgd_iters', type=int, required=True)
@click.option('--batch_size', type=int, required=True)
@click.option('--pretrain_epochs', type=int, required=True)
@click.option('--lr', type=float, required=True)
@click.option('--swa_epochs', type=int, required=True)
@click.option('--swa_lr', type=float, required=True, multiple=True)
def cli(adv_epsilon, pgd_iters, batch_size, pretrain_epochs, lr, swa_epochs, swa_lr):
    device_count = jax.local_device_count()
    assert batch_size % device_count == 0, f'batch_size should be divisible by {device_count}'

    assert len(set(swa_lr)) == len(swa_lr) > 0, 'swa_lr should consist of unique learning rates'

    root = '/home/qys/torchvision/datasets'
    specimen = jnp.empty((32, 32, 3))

    mean_rgb = np.array((0.4914, 0.4822, 0.4465))
    std_rgb = np.array((0.247, 0.243, 0.261))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean_rgb, std_rgb),
        lambda X: torch.permute(X, (1, 2, 0)),
    ])

    train_dataset = CIFAR10(root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root, train=False, download=True, transform=transform)

    key = jax.random.PRNGKey(42)
    key_init, key = jax.random.split(key)
    state_pretrain = create_train_state(key_init, 10, lr, specimen)
    state_pretrain = replicate(state_pretrain)


    # Adversarial Training with PGD
    print('===> Pretraining')
    train_loader = DataLoader(train_dataset, batch_size)
    for epoch in range(pretrain_epochs):
        key, state_pretrain, loss = train_epoch(key, state_pretrain, device_count, adv_epsilon, pgd_iters, train_loader)
        with jnp.printoptions(precision=3):
            print(f'Epoch {epoch + 1}, train loss: {loss}')


    print('===> SWA')
    ingredients = {}
    for lr in swa_lr:
        print(f'---> Finetuning with lr = {lr}')
        state_finetune = create_finetune_state(unreplicate(state_pretrain), lr)
        state_swa = state_finetune = replicate(state_finetune)
        for epoch in range(swa_epochs):
            key, state_finetune, loss = train_epoch(key, state_finetune, device_count, adv_epsilon, pgd_iters, train_loader)
            with jnp.printoptions(precision=3):
                print(f'Epoch {epoch + 1}, finetune loss: {loss}')

            w = 1 / (epoch+1)
            state_swa = state_swa.replace(
                params = jax.tree_util.tree_map(lambda x, y: (1-w) * x + w * y, state_swa.params, state_finetune.params)
            )

        state_swa = update_batch_stats_epoch(key, state_swa, device_count, adv_epsilon, pgd_iters, train_loader)

        ingredients[lr] = (state_swa, state_finetune)


    print('===> Making soups')
    # TODO: greedy soup
    state_mega_soup = state_soup = state_pretrain
    for i, (state_swa, state_finetune) in enumerate(ingredients.values()):
        w = 1 / (i+1)
        state_mega_soup = state_mega_soup.replace(
            params = jax.tree_util.tree_map(lambda x, y: (1-w) * x + w * y, state_mega_soup.params, state_swa.params)
        )
        state_soup = state_soup.replace(
            params = jax.tree_util.tree_map(lambda x, y: (1-w) * x + w * y, state_soup.params, state_finetune.params)
        )
    
    state_mega_soup = update_batch_stats_epoch(key, state_mega_soup, device_count, adv_epsilon, pgd_iters, train_loader)
    state_soup = update_batch_stats_epoch(key, state_soup, device_count, adv_epsilon, pgd_iters, train_loader)


    print('===> Evaluating adversarial accuracy')
    states = [
        ('MEGASoup', state_mega_soup),
        ('Soup', state_soup),
        ('Pretrain', state_pretrain),
    ]
    for lr, (state_swa, state_finetune) in ingredients.items():
        states.append((f'SWA-{lr}', state_swa))
        states.append((f'Finetune-{lr}', state_finetune))

    for dataset_name, dataset in [
            ('Test', test_dataset),
            ('Train', train_dataset)]:
        for state_name, state in states:
            total_hits_orig, total_hits_adv = 0, 0
            loader = DataLoader(dataset, batch_size)
            for X, y in loader:
                image = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
                label = jnp.array(y).reshape(device_count, -1, *y.shape[1:])

                *key_attack, key = jax.random.split(key, device_count + 1)
                key_attack = jnp.array(key_attack)
                image_adv = pgd_untargeted(key_attack, state, image, label, adv_epsilon/255, pgd_iters)

                total_hits_orig += test_step(state, image, label).sum()
                total_hits_adv += test_step(state, image_adv, label).sum()

            total = len(dataset)
            accuracy_orig = total_hits_orig/total*100
            accuracy_adv = total_hits_adv/total*100
            print(f'[{dataset_name}] {state_name} accuracy: natural {accuracy_orig:.2f}%, adv {accuracy_adv:.2f}%')


def train_epoch(key, state, device_count, adv_epsilon, pgd_iters, train_loader):
    epoch_predictive = 0
    for X, y in train_loader:
        image = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        label = jnp.array(y).reshape(device_count, -1, *y.shape[1:])

        *key_attack, key = jax.random.split(key, device_count + 1)
        key_attack = jnp.array(key_attack)
        image_adv = pgd_untargeted(key_attack, state, image, label, adv_epsilon/255, pgd_iters)

        state, predictive, _ = train_step(state, image, image_adv, label)
        epoch_predictive += predictive.sum()

    state = state.replace(batch_stats=cross_replica_mean(state.batch_stats))

    return key, state, epoch_predictive


def update_batch_stats_epoch(key, state, device_count, adv_epsilon, pgd_iters, train_loader):
    # Update SWA batch normalization statistics at the end of training
    for X, y in train_loader:
        image = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        label = jnp.array(y).reshape(device_count, -1, *y.shape[1:])

        *key_attack, key = jax.random.split(key, device_count + 1)
        key_attack = jnp.array(key_attack)
        image_adv = pgd_untargeted(key_attack, state, image, label, adv_epsilon/255, pgd_iters)

        # TODO: should we feed in natural images instead?
        state = update_batch_stats_step(state, image_adv)

    state = state.replace(batch_stats=cross_replica_mean(state.batch_stats))

    return state


if __name__ == '__main__':
    initialize_cache('jit_cache')
    cli()

