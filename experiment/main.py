import jax
import jax.numpy as jnp
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm

from .train import create_train_state, train_step, test_step
from .distances import avg_distance
from .attacks import pgd_attack


def run(learning_rate, num_epochs, batch_size):
    root = "/usr/local/share/torchvision/datasets"
    mean_rgb = np.array((0.4914, 0.4822, 0.4465))
    std_rgb = np.array((0.247, 0.243, 0.261))
    specimen = jnp.empty((32, 32, 3))

    def transform(x):
        return ((np.array(x) / 255. - mean_rgb) / std_rgb).reshape(specimen.shape)

    train_dataset = CIFAR10(root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root, train=False, download=True, transform=transform)

    key = jax.random.PRNGKey(42)
    state = create_train_state(key, 10, learning_rate, specimen)

    for _ in range(num_epochs):
        train_loader = DataLoader(train_dataset, batch_size)
        pbar = tqdm(train_loader)
        for X, y in pbar:
            image = jnp.array(X)
            label = jnp.array(y)
            state, loss = train_step(state, image, label)
            pbar.set_description(f"{loss.item()=:.2f}")

        distances = avg_distance(state.centroids, state.centroids, jnp.arange(state.centroids.shape[0]))
        print("avg_dist(state.centroids)", distances.round(2))
        print("norm(state.centroids)", jnp.linalg.norm(state.centroids, axis=-1))

    # Construct adversial examples with PGD for the test set
    test_loader = DataLoader(test_dataset, 1024)

    total_hits_orig, total_hits_adv = 0, 0
    for X, y in test_loader:
        image = jnp.array(X)
        label = jnp.array(y)

        target = jax.nn.one_hot((label + 1) % 10, 10)
        adversary = pgd_attack(image, target, state)

        total_hits_orig += test_step(state, image, label)
        total_hits_adv += test_step(state, adversary, label)

    total = len(test_dataset)
    accuracy_orig = total_hits_orig/total*100
    accuracy_adv = total_hits_adv/total*100
    print(f"Accuracy: {accuracy_orig:.2f}% -> {accuracy_adv:.2f}%")


if __name__ == "__main__":
    learning_rate = 1e-2
    num_epochs = 8
    batch_size = 256
    run(learning_rate, num_epochs, batch_size)

