import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache.compilation_cache import initialize_cache
from flax.jax_utils import replicate
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import click

from .train import create_learning_rate, create_train_state, train_step, update_batch_stats_step, cross_replica_mean, test_step
from .attacks import pgd_untargeted


@click.command()
@click.option('--adv_epsilon', type=float, required=True)
@click.option('--pgd_iters', type=int, required=True)
@click.option('--batch_size', type=int, required=True)
@click.option('--epochs', type=int, required=True)
@click.option('--lr', type=float, required=True)
@click.option('--swa_start', type=int, required=True)
@click.option('--swa_lr', type=float, required=True)
def cli(adv_epsilon, pgd_iters, lr, batch_size, epochs, swa_lr, swa_start):
    device_count = jax.local_device_count()
    assert batch_size % device_count == 0, f'batch_size is not divisible by {device_count}'

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
    steps_per_epoch = (len(train_dataset) + batch_size - 1) // batch_size
    learning_rate = create_learning_rate(lr, swa_lr, swa_start, steps_per_epoch)
    state_last = create_train_state(key_init, 10, learning_rate, specimen)
    state_swa = state_last = replicate(state_last)

    # Adversarial Training with PGD
    train_loader = DataLoader(train_dataset, batch_size)

    for epoch in range(epochs):
        epoch_predictive = 0
        epoch_consistency = 0
        for X, y in train_loader:
            image = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
            label = jnp.array(y).reshape(device_count, -1, *y.shape[1:])

            *key_attack, key = jax.random.split(key, device_count + 1)
            key_attack = jnp.array(key_attack)
            image_adv = pgd_untargeted(key_attack, state_last, image, label, adv_epsilon/255, pgd_iters)

            state_last, predictive, consistency = train_step(state_last, image, image_adv, label)
            epoch_predictive += predictive.sum()
            epoch_consistency += consistency.sum()

        state_last = state_last.replace(batch_stats=cross_replica_mean(state_last.batch_stats))

        if epoch >= swa_start:
            w = 1 / (epoch-swa_start+1)
            state_swa = jax.tree_util.tree_map(lambda x, y: (1-w) * x + w * y, state_swa, state_last)

        with jnp.printoptions(precision=3):
            print(f'===> Epoch {epoch + 1}, train predictive: {epoch_predictive}, train consistency: {epoch_consistency}')


    # Update SWA batch normalization statistics at the end of training
    for X, y in train_loader:
        image = jnp.array(X).reshape(device_count, -1, *X.shape[1:])
        label = jnp.array(y).reshape(device_count, -1, *y.shape[1:])

        *key_attack, key = jax.random.split(key, device_count + 1)
        key_attack = jnp.array(key_attack)
        image_adv = pgd_untargeted(key_attack, state_swa, image, label, adv_epsilon/255, pgd_iters)

        # TODO: should we feed in natural images instead?
        state_swa = update_batch_stats_step(state_swa, image_adv)

    state_swa = state_swa.replace(batch_stats=cross_replica_mean(state_swa.batch_stats))


    # Evaluate adversarial accuracy
    for state_name, state in [
            ('SWA', state_swa),
            ('Last', state_last)]:
        for dataset_name, dataset in [
                ('Train', train_dataset),
                ('Test', test_dataset)]:
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
            print(f'[{state_name}] {dataset_name} accuracy: natural {accuracy_orig:.2f}%, adv {accuracy_adv:.2f}%')


if __name__ == '__main__':
    initialize_cache('jit_cache')
    cli()

