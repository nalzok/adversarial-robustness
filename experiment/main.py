import jax
import jax.numpy as jnp
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from .train import create_train_state, train_step, test_step
from .centroids import avg_dist
from .attack import pgd_attack


def run(learning_rate, num_epochs, batch_size):
    root = "/usr/local/share/torchvision/datasets"
    train_dataset = CIFAR10(root, train=True, download=True, transform=np.array)
    test_dataset = CIFAR10(root, train=False, download=True, transform=np.array)
    specimen = jnp.empty((32, 32, 3))

    key = jax.random.PRNGKey(42)
    state = create_train_state(key, 10, learning_rate, specimen)

    for _ in range(num_epochs):
        train_loader = DataLoader(train_dataset, batch_size)
        pbar = tqdm(train_loader)
        for X, y in pbar:
            image = jnp.array(X).reshape((-1, *specimen.shape)) / 255.0
            label = jnp.array(y)
            state, loss = train_step(state, image, label)
            pbar.set_description(f"{loss.item()=:.2f}")

        print("avg_dist(state.centroids)", avg_dist(state.centroids, state.centroids).round(2))

    # Construct adversial examples with PGD for the test set
    exported = set()
    test_loader = DataLoader(test_dataset, 1024)

    total_hits_orig, total_hits_adv = 0, 0
    for i, (X, y) in enumerate(test_loader):
        image = jnp.array(X).reshape((-1, *specimen.shape)) / 255.0
        label = jnp.array(y)

        target = jax.nn.one_hot((label + 1) % 10, 10)
        adversary = pgd_attack(image, target, state)

        # for i in range(label.shape[0]):
        #     single_label = label[i].item()
        #     if single_label not in exported:
        #         exported.add(single_label)
        #
        #         fig, axes = plt.subplots(1, 2, constrained_layout=True)
        #         axes[0].imshow(image[i])
        #         axes[1].imshow(adversary[i])
        #         for ax in axes:
        #             ax.get_xaxis().set_visible(False)
        #             ax.get_yaxis().set_visible(False)
        #         fig.suptitle(
        #             f"Label = {single_label}, Target = {(single_label + 1) % 10}"
        #         )
        #         fig.savefig(f"plots/adversarial_{single_label}.png", dpi=200)
        #         plt.close()

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

