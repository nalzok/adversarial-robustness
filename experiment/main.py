import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

from .train import create_train_state, train_step, test_step
from .attacks import pgd_untargeted


def run(learning_rate, num_epochs, batch_size):
    root = "/usr/local/share/torchvision/datasets"
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
    state = create_train_state(key_init, 10, learning_rate, specimen)

    # Adversarial Training with PGD
    for epoch in range(num_epochs):
        train_loader = DataLoader(train_dataset, batch_size)
        epoch_loss = 0
        for X, y in train_loader:
            image = jnp.array(X)
            label = jnp.array(y)

            key_attack, key = jax.random.split(key)
            image_adv = pgd_untargeted(key_attack, state, image, label,
                    epsilon=16/255, max_steps=2000, step_size=1/255, randomize=True)

            state, loss = train_step(state, image, image_adv, label)
            epoch_loss += loss

        with jnp.printoptions(precision=3):
            print(f"===> Epoch {epoch + 1}, train loss: {epoch_loss}")

    # Evaluate adversarial accuracy
    test_loader = DataLoader(test_dataset, batch_size)

    total_hits_orig, total_hits_adv = 0, 0
    for X, y in test_loader:
        image = jnp.array(X)
        label = jnp.array(y)

        key_attack, key = jax.random.split(key)
        adversary = pgd_untargeted(key_attack, state, image, label,
                epsilon=16/255, max_steps=2000, step_size=1/255, randomize=False)

        total_hits_orig += test_step(state, image, label)
        total_hits_adv += test_step(state, adversary, label)

    total = len(test_dataset)
    accuracy_orig = total_hits_orig/total*100
    accuracy_adv = total_hits_adv/total*100
    print(f"Accuracy: {accuracy_orig:.2f}% -> {accuracy_adv:.2f}%")


if __name__ == "__main__":
    learning_rate = 1e-3
    num_epochs = 8
    batch_size = 256
    run(learning_rate, num_epochs, batch_size)

