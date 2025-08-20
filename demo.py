# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# ff_forward_only.py
import argparse
import math
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms
import tt_torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import BackendOptions


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rms_layer_norm(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Normalize each sample vector by its RMS (no learnable params)
    return z / (torch.sqrt(torch.mean(z**2, dim=-1, keepdim=True)) + eps)


def one_hot(
    idx: int, num_classes: int = 10, device: torch.device = torch.device("cpu")
):
    v = torch.zeros(num_classes, device=device)
    v[idx] = 1.0
    return v


# -----------------------------
# Dataset: MNIST with label-embedding for FF
# -----------------------------
class FFMNIST(data.Dataset):
    """
    Returns (inputs, labels) where:
      - inputs is a dict with:
          pos_images:   batch of images with 1-hot label embedded in row 0, cols 0..9 (true label)
          neg_images:   same but with a wrong label embedded
          neutral:      same but with uniform label (1/10) embedded
      - labels is a dict with:
          class_labels: integer class label
    Shapes:
      - all images are [1, 28, 28] float32 in [0,1]
    """

    def __init__(self, root, train, device, num_classes=10, download=True):
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            transform=transforms.ToTensor(),
            download=download,
        )
        self.device = device
        self.num_classes = num_classes
        self.uniform = torch.ones(num_classes, device=device) / num_classes

    def __len__(self):
        return len(self.dataset)

    def _embed_label(self, img, label_vec):
        # img: [1,28,28]; label_vec: [10]
        out = img.clone()
        out[0, 0, : self.num_classes] = label_vec
        return out

    def __getitem__(self, idx):
        img, y = self.dataset[idx]
        img = img.to(self.device)
        # Positive = true label
        pos = self._embed_label(img, one_hot(y, self.num_classes, self.device))
        # Negative = random incorrect label
        wrong_choices = list(range(self.num_classes))
        wrong_choices.remove(y)
        neg_y = random.choice(wrong_choices)
        neg = self._embed_label(img, one_hot(neg_y, self.num_classes, self.device))
        # Neutral = uniform label
        neu = self._embed_label(img, self.uniform)

        inputs = {
            "pos_images": pos,
            "neg_images": neg,
            "neutral_sample": neu,
        }
        labels = {"class_labels": torch.tensor(y, device=self.device, dtype=torch.long)}
        return inputs, labels


# -----------------------------
# Model
# -----------------------------
@dataclass
class FFConfig:
    hidden_dim: int = 1024
    num_layers: int = 2
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 5
    momentum: float = 0.0  # if you later add peer-normalization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


class ForwardOnlyFF(nn.Module):
    """
    A Forward-Forward MLP trained with *manual* local updates (no autograd).
    - Hidden layers: Linear -> ReLU (forward), but gradient ignores ReLU mask (full-grad)
    - FF loss per layer: BCEWithLogits(E - d, target), E=sum(z^2), d=layer width
    - Classifier: linear (no bias) trained with manual softmax-CE grads on neutral features
    """

    def __init__(self, cfg: FFConfig):
        super().__init__()
        self.cfg = cfg
        self.act = torch.relu  # forward only; gradient is manual and ignores mask
        dims = [784] + [cfg.hidden_dim] * cfg.num_layers
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(cfg.num_layers)]
        )
        # init like repo: N(0, 1/sqrt(fan_in))
        for lin in self.layers:
            nn.init.normal_(
                lin.weight, mean=0.0, std=1.0 / math.sqrt(lin.weight.shape[0])
            )
            nn.init.zeros_(lin.bias)

        # Classifier on top of concatenated features from layers 1..L-1 (like the repo)
        concat_dim = sum([cfg.hidden_dim for _ in range(max(cfg.num_layers - 1, 0))])
        self.classifier = nn.Linear(concat_dim, 10, bias=False)
        nn.init.zeros_(self.classifier.weight)

        # Turn off autograd on all params (we will update .data manually)
        for p in self.parameters():
            p.requires_grad_(False)

    # ---------- Forward helpers (no grad tracking) ----------
    def forward_hidden_no_grad(self, x_flat: torch.Tensor):
        """
        Forward through hidden layers with RMS layer-norm after each layer.
        Returns list of layer activations [z0, z1, ...] with shape [B, hidden_dim].
        """
        z_list = []
        z = x_flat
        for lin in self.layers:
            z = self.act(z @ lin.weight.t() + lin.bias)
            z = rms_layer_norm(z)
            z_list.append(z)
        return z_list

    def forward_features_for_classifier(self, x_flat: torch.Tensor):
        """
        Compute feature concat for classifier: concat z from layers [1..L-1].
        """
        z_list = self.forward_hidden_no_grad(x_flat)
        if len(z_list) <= 1:
            return torch.zeros(x_flat.size(0), 0, device=x_flat.device)
        feats = torch.cat(z_list[1:], dim=-1)
        return feats

    # ---------- Manual update rules ----------
    @staticmethod
    def _ff_layer_update_inplace(
        lin: nn.Linear, x: torch.Tensor, labels01: torch.Tensor, lr: float
    ):
        """
        Manual local update for a single layer using FF loss.
        - Forward: z = ReLU(Wx + b)
        - Energy:  E = sum(z^2)
        - Loss:    BCEWithLogits(E - d, y)
        - Gradient wrt W: ((sigmoid(E-d)-y)*2*z) ⊗ x  (averaged over batch)
          NOTE: ignores ReLU derivative mask (full-grad).
        """
        # forward
        pre = x @ lin.weight.data.t() + lin.bias.data  # [B, out]
        z = torch.clamp(pre, min=0.0)  # ReLU forward
        d = z.shape[1]
        energy = torch.sum(z * z, dim=1)  # [B]
        logits = energy - d
        probs = torch.sigmoid(logits)  # [B]

        # dLoss/dE = sigmoid(E-d) - y  (BCEWithLogits derivative)
        dE = probs - labels01  # [B]
        # dE/dz = 2z => dLoss/dz = (sigmoid(E-d)-y)*2*z
        dZ = (2.0 * z) * dE[:, None]  # [B, out]

        # grads wrt W, b (average over batch)
        B = x.size(0)
        dW = (dZ.t() @ x) / B  # [out, in]
        db = dZ.mean(dim=0)  # [out]

        # SGD update
        lin.weight.data -= lr * dW
        lin.bias.data -= lr * db

        # Return post-activation (detached) as next layer input (normalized)
        z_next = rms_layer_norm(z)
        return z_next, probs.detach()

    @staticmethod
    def _classifier_update_inplace(
        lin: nn.Linear, X: torch.Tensor, y_long: torch.Tensor, lr: float
    ):
        """
        Manual CE update for linear classifier (no bias).
        logits = X W^T;  softmax;  CE
        grad dL/dW = ( (softmax(logits) - one_hot(y))^T @ X ) / B
        """
        if X.numel() == 0:
            return 0.0  # nothing to train if only 1 hidden layer

        logits = X @ lin.weight.data.t()  # [B, C]
        logits = logits - logits.max(dim=1, keepdim=True)[0]  # stability
        exp_logits = torch.exp(logits)
        probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)  # [B, C]

        B, C = probs.shape
        T = torch.zeros_like(probs)
        T[torch.arange(B, device=X.device), y_long] = 1.0

        dY = (probs - T) / B  # [B, C]
        dW = dY.t() @ X  # [C, D]
        lin.weight.data -= lr * dW

        preds = probs.argmax(dim=1)
        acc = (preds == y_long).float().mean().item()
        return acc

    # ---------- Training steps (no autograd) ----------
    def ff_train_step(
        self, pos_batch: torch.Tensor, neg_batch: torch.Tensor, lr: float
    ):
        """
        One FF training step over concatenated (pos,neg).
        pos_batch/neg_batch: [B, 1, 28, 28]
        """
        B = pos_batch.size(0)
        x = torch.cat([pos_batch, neg_batch], dim=0)  # [2B,1,28,28]
        labels01 = torch.zeros(2 * B, device=x.device)
        labels01[:B] = 1.0  # pos=1, neg=0
        x = x.view(2 * B, -1)  # flatten
        x = rms_layer_norm(x)

        # Train each FF layer locally, pass normalized z forward
        z = x
        all_probs = []
        for lin in self.layers:
            z, probs = self._ff_layer_update_inplace(lin, z, labels01, lr=lr)
            all_probs.append(probs)

        # Optional: return mean FF accuracies per layer
        ff_accs = []
        for probs in all_probs:
            preds = (probs > 0.5).float()
            acc = (preds == labels01).float().mean().item()
            ff_accs.append(acc)
        return ff_accs

    def clf_train_step(
        self, neutral_batch: torch.Tensor, labels_long: torch.Tensor, lr: float
    ):
        """
        Train the linear classifier on neutral features (no autograd).
        """
        x = neutral_batch.view(neutral_batch.size(0), -1)
        x = rms_layer_norm(x)
        feats = self.forward_features_for_classifier(x)
        acc = self._classifier_update_inplace(
            self.classifier, feats, labels_long, lr=lr
        )
        return acc

    def compile_forward(self, backend="tt"):
        # Compile the main forward functions
        cc = CompilerConfig()
        cc.enable_consteval = True
        cc.consteval_parameters = True

        options = BackendOptions()
        options.compiler_config = cc
        self.forward_hidden_no_grad = torch.compile(
            self.forward_hidden_no_grad,
            backend=backend,
            fullgraph=True,
            options=options,
        )
        self.forward_features_for_classifier = torch.compile(
            self.forward_features_for_classifier,
            backend=backend,
            fullgraph=True,
            options=options,
        )


# -----------------------------
# Training loop
# -----------------------------
def train_forward_only(cfg: FFConfig):
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    # Data
    train_ds = FFMNIST(root="./data", train=True, device=device, download=True)
    test_ds = FFMNIST(root="./data", train=False, device=device, download=True)

    def collate(batch):
        # batch: list of (inputs, labels)
        # Stack pos/neg/neutral and labels
        pos = torch.stack([b[0]["pos_images"] for b in batch], dim=0)
        neg = torch.stack([b[0]["neg_images"] for b in batch], dim=0)
        neu = torch.stack([b[0]["neutral_sample"] for b in batch], dim=0)
        y = torch.stack([b[1]["class_labels"] for b in batch], dim=0)
        return {"pos": pos, "neg": neg, "neu": neu, "y": y}

    train_loader = data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
    )
    test_loader = data.DataLoader(
        test_ds, batch_size=1024, shuffle=False, drop_last=False, collate_fn=collate
    )

    # Model
    model = ForwardOnlyFF(cfg).to(device)
    model.compile_forward()

    # Train
    for epoch in range(1, cfg.epochs + 1):
        ff_acc_meter = [0.0 for _ in range(cfg.num_layers)]
        ff_count = 0
        clf_acc_meter = 0.0
        clf_count = 0

        model.train()
        with torch.no_grad():
            for batch in train_loader:
                pos = batch["pos"]
                neg = batch["neg"]
                neu = batch["neu"]
                y = batch["y"]

                # FF layer training step
                ff_accs = model.ff_train_step(pos, neg, lr=cfg.lr)
                for i, a in enumerate(ff_accs):
                    ff_acc_meter[i] += a
                ff_count += 1

                # Classifier training step
                acc = model.clf_train_step(neu, y, lr=cfg.lr)
                clf_acc_meter += acc
                clf_count += 1

        ff_acc_report = ", ".join(
            [
                f"L{i}:{ff_acc_meter[i]/max(ff_count,1):.3f}"
                for i in range(cfg.num_layers)
            ]
        )
        print(
            f"[Epoch {epoch:02d}] FF accs [{ff_acc_report}] | Clf acc: {clf_acc_meter/max(clf_count,1):.3f}"
        )

        # Simple evaluation
        if epoch % 1 == 0:
            test_acc = evaluate(model, test_loader)
            print(f"          Test classifier acc: {test_acc:.3f}")
    print("Saving model dict...")
    torch.save(model.state_dict(), "ff_forward_only.pt")
    return model


def evaluate(model: ForwardOnlyFF, loader: data.DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            neu = batch["neu"]
            y = batch["y"]
            x = neu.view(neu.size(0), -1)
            x = rms_layer_norm(x)
            feats = model.forward_features_for_classifier(x)
            if feats.numel() == 0:
                # No features if model has only 1 layer
                logits = torch.zeros(neu.size(0), 10, device=neu.device)
            else:
                logits = feats @ model.classifier.weight.t()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def predict(model: ForwardOnlyFF, image: torch.Tensor) -> int:
    """
    image: [1,28,28] tensor in [0,1], single MNIST digit.
    Returns predicted class index.
    """
    model.eval()
    with torch.no_grad():
        # Embed neutral label into image
        img = image.clone()
        img[0, 0, :10] = 0.1  # uniform label embedding

        # Flatten + normalize
        x = img.view(1, -1)
        x = rms_layer_norm(x)

        # Extract features
        feats = model.forward_features_for_classifier(x)

        # Classifier logits
        if feats.numel() == 0:
            logits = torch.zeros(1, 10, device=x.device)
        else:
            logits = feats @ model.classifier.weight.data.t()

        pred = logits.argmax(dim=1).item()
        return pred


# -----------------------------
# Inference
# -----------------------------


def display_results(image, label, pred, filename="result.png"):
    """Simple display of image and prediction with save"""
    plt.figure(figsize=(6, 4))
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Predicted: {pred}, Ground truth: {label}")
    plt.axis("off")
    plt.savefig(filename)
    plt.show()


def inference(seed: int):
    cfg = FFConfig(hidden_dim=1024, num_layers=2, device="cpu")
    model = ForwardOnlyFF(cfg)
    model.load_state_dict(torch.load("ff_forward_only.pt", map_location=cfg.device))
    model.eval()
    test_ds = datasets.MNIST(
        "./data", train=False, transform=transforms.ToTensor(), download=True
    )
    img, label = test_ds[seed]
    pred = predict(model, img)
    display_results(img, label, pred)


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = FFConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    model = train_forward_only(cfg)
    # How to call inference loop:
    # inference(4)
