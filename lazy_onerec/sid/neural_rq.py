"""Self-contained RQ-VAE and RQ-Kmeans+ components adapted from MiniOneRec."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from .distance import validate_distance_metric


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install PyTorch to use RQ-VAE/RQ-Kmeans+") from exc
    return torch


def _kmeans_centers(
    values,
    n_clusters: int,
    n_iters: int,
    distance_metric: str,
):
    torch = _torch()
    from sklearn.cluster import KMeans

    fit_values = values
    if distance_metric == "cosine":
        fit_values = torch.nn.functional.normalize(values, dim=-1)
    centers = KMeans(
        n_clusters=n_clusters,
        max_iter=n_iters,
        n_init=10,
        random_state=42,
    ).fit(fit_values.detach().cpu().numpy()).cluster_centers_
    result = values.new_tensor(centers)
    if distance_metric == "cosine":
        result = torch.nn.functional.normalize(result, dim=-1)
    return result


def _sinkhorn(distances, epsilon: float, iterations: int):
    torch = _torch()
    assignment = torch.exp(-distances / epsilon)
    assignment /= assignment.sum().clamp_min(1e-12)
    batch, clusters = assignment.shape
    for _ in range(iterations):
        assignment /= assignment.sum(dim=1, keepdim=True).clamp_min(1e-12)
        assignment /= batch
        assignment /= assignment.sum(dim=0, keepdim=True).clamp_min(1e-12)
        assignment /= clusters
    return assignment * batch


class MLPLayers:
    """Factory kept separate so importing artifact utilities needs no PyTorch."""

    @staticmethod
    def build(dims: Sequence[int], dropout: float = 0.0, bn: bool = False):
        torch = _torch()
        nn = torch.nn
        modules = []
        for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            modules.append(nn.Dropout(dropout))
            linear = nn.Linear(input_dim, output_dim)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            modules.append(linear)
            if index != len(dims) - 2:
                if bn:
                    modules.append(nn.BatchNorm1d(output_dim))
                modules.append(nn.ReLU())
        return nn.Sequential(*modules)


def build_rqvae_class():
    torch = _torch()
    nn = torch.nn
    functional = torch.nn.functional

    class VectorQuantizer(nn.Module):
        def __init__(
            self,
            n_codes: int,
            dim: int,
            beta: float,
            kmeans_init: bool,
            kmeans_iters: int,
            sinkhorn_epsilon: float,
            sinkhorn_iters: int,
            distance_metric: str,
        ):
            super().__init__()
            self.n_codes = n_codes
            self.dim = dim
            self.beta = beta
            self.kmeans_iters = kmeans_iters
            self.sinkhorn_epsilon = sinkhorn_epsilon
            self.sinkhorn_iters = sinkhorn_iters
            self.distance_metric = validate_distance_metric(distance_metric)
            self.embedding = nn.Embedding(n_codes, dim)
            self.register_buffer(
                "initialized", torch.tensor(not kmeans_init, dtype=torch.bool)
            )
            if kmeans_init:
                nn.init.zeros_(self.embedding.weight)
            else:
                nn.init.uniform_(
                    self.embedding.weight, -1.0 / n_codes, 1.0 / n_codes
                )

        def forward(self, values, use_sinkhorn: bool = True):
            flat = values.reshape(-1, self.dim)
            if not bool(self.initialized) and self.training:
                if flat.shape[0] < self.n_codes:
                    raise ValueError(
                        "first RQ-VAE batch must contain at least codebook_size rows"
                    )
                centers = _kmeans_centers(
                    flat,
                    self.n_codes,
                    self.kmeans_iters,
                    self.distance_metric,
                )
                self.embedding.weight.data.copy_(centers)
                self.initialized.fill_(True)

            if self.distance_metric == "cosine":
                normalized_values = functional.normalize(flat, dim=-1)
                normalized_codes = functional.normalize(
                    self.embedding.weight, dim=-1
                )
                distances = 1.0 - normalized_values @ normalized_codes.t()
            else:
                distances = (
                    flat.square().sum(dim=1, keepdim=True)
                    + self.embedding.weight.square().sum(dim=1).unsqueeze(0)
                    - 2 * flat @ self.embedding.weight.t()
                )
            if use_sinkhorn and self.sinkhorn_epsilon > 0:
                centered = distances - distances.mean()
                scale = centered.abs().max().clamp_min(1e-5)
                indices = _sinkhorn(
                    (centered / scale).double(),
                    self.sinkhorn_epsilon,
                    self.sinkhorn_iters,
                ).argmax(dim=-1)
            else:
                indices = distances.argmin(dim=-1)

            quantized = self.embedding(indices).view_as(values)
            commitment = functional.mse_loss(quantized.detach(), values)
            codebook = functional.mse_loss(quantized, values.detach())
            loss = codebook + self.beta * commitment
            straight_through = values + (quantized - values).detach()
            return straight_through, loss, indices.view(values.shape[:-1])

    class ResidualVectorQuantizer(nn.Module):
        def __init__(
            self,
            codebook_sizes,
            dim,
            beta,
            kmeans_init,
            kmeans_iters,
            sinkhorn_epsilons,
            sinkhorn_iters,
            distance_metric,
        ):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    VectorQuantizer(
                        size,
                        dim,
                        beta,
                        kmeans_init,
                        kmeans_iters,
                        epsilon,
                        sinkhorn_iters,
                        distance_metric,
                    )
                    for size, epsilon in zip(
                        codebook_sizes, sinkhorn_epsilons
                    )
                ]
            )

        def forward(self, values, use_sinkhorn=True):
            residual = values
            quantized_sum = torch.zeros_like(values)
            losses, indices = [], []
            for layer in self.layers:
                quantized, loss, level_indices = layer(
                    residual, use_sinkhorn=use_sinkhorn
                )
                residual = residual - quantized
                quantized_sum = quantized_sum + quantized
                losses.append(loss)
                indices.append(level_indices)
            return (
                quantized_sum,
                torch.stack(losses).mean(),
                torch.stack(indices, dim=-1),
            )

    class RQVAE(nn.Module):
        def __init__(
            self,
            input_dim: int,
            codebook_sizes: Sequence[int],
            latent_dim: int = 32,
            hidden_dims: Sequence[int] = (512, 256, 128),
            dropout: float = 0.0,
            bn: bool = False,
            beta: float = 0.25,
            quant_loss_weight: float = 1.0,
            kmeans_init: bool = True,
            kmeans_iters: int = 100,
            sinkhorn_epsilons: Sequence[float] | None = None,
            sinkhorn_iters: int = 50,
            distance_metric: str = "euclidean",
        ):
            super().__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.codebook_sizes = tuple(codebook_sizes)
            self.quant_loss_weight = quant_loss_weight
            sinkhorn_epsilons = tuple(
                sinkhorn_epsilons or [0.0] * len(codebook_sizes)
            )
            if len(sinkhorn_epsilons) != len(codebook_sizes):
                raise ValueError("one sinkhorn epsilon is required per level")

            encoder_dims = [input_dim, *hidden_dims, latent_dim]
            decoder_dims = list(reversed(encoder_dims))
            self.encoder = MLPLayers.build(encoder_dims, dropout, bn)
            self.quantizer = ResidualVectorQuantizer(
                codebook_sizes,
                latent_dim,
                beta,
                kmeans_init,
                kmeans_iters,
                sinkhorn_epsilons,
                sinkhorn_iters,
                distance_metric,
            )
            self.decoder = MLPLayers.build(decoder_dims, dropout, bn)

        def forward(self, values, use_sinkhorn=True):
            encoded = self.encoder(values)
            quantized, quant_loss, indices = self.quantizer(
                encoded, use_sinkhorn=use_sinkhorn
            )
            return self.decoder(quantized), quant_loss, indices

        def loss(self, reconstructed, quant_loss, values):
            reconstruction = functional.mse_loss(reconstructed, values)
            return (
                reconstruction + self.quant_loss_weight * quant_loss,
                reconstruction,
            )

        @torch.no_grad()
        def get_indices(self, values):
            encoded = self.encoder(values)
            _, _, indices = self.quantizer(
                encoded, use_sinkhorn=False
            )
            return indices

        def codebooks(self):
            return [
                layer.embedding.weight.detach().cpu().numpy().copy()
                for layer in self.quantizer.layers
            ]

    return RQVAE


@dataclass
class NeuralTrainResult:
    model: object
    codes: np.ndarray
    codebooks: List[np.ndarray]
    best_loss: float
    collision_rate: float


def make_rq_kmeans_plus(
    model,
    codebooks: Sequence[np.ndarray],
) -> None:
    """Apply MiniOneRec RQ-Kmeans+ identity encoder and codebook warm start."""
    torch = _torch()
    nn = torch.nn

    class ResidualEncoder(nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.mlp = mlp

        def forward(self, values):
            return values + self.mlp(values)

    if model.input_dim != model.latent_dim:
        raise ValueError("RQ-Kmeans+ requires latent_dim == input_dim")
    model.encoder = ResidualEncoder(model.encoder)
    last_linear = next(
        module
        for module in reversed(list(model.encoder.mlp.modules()))
        if isinstance(module, nn.Linear)
    )
    nn.init.zeros_(last_linear.weight)
    nn.init.zeros_(last_linear.bias)

    if len(codebooks) != len(model.quantizer.layers):
        raise ValueError("warm-start codebook count does not match RQ levels")
    with torch.no_grad():
        for layer, values in zip(model.quantizer.layers, codebooks):
            tensor = torch.as_tensor(
                values,
                dtype=layer.embedding.weight.dtype,
                device=layer.embedding.weight.device,
            )
            if tensor.shape != layer.embedding.weight.shape:
                raise ValueError(
                    f"warm-start shape {tuple(tensor.shape)} != "
                    f"{tuple(layer.embedding.weight.shape)}"
                )
            layer.embedding.weight.copy_(tensor)
            layer.initialized.fill_(True)


def train_neural_rq(
    embeddings: np.ndarray,
    model,
    device: str = "cuda",
    epochs: int = 500,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    warmup_epochs: int = 10,
    eval_every: int = 10,
    seed: int = 2024,
) -> NeuralTrainResult:
    torch = _torch()
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    tensor = torch.from_numpy(
        embeddings.astype(np.float32, copy=False)
    )
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.startswith("cuda"),
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    warmup_steps = max(1, warmup_epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup_steps),
    )

    best_key = (float("inf"), float("inf"))
    best_state: Dict[str, object] | None = None
    best_loss = float("inf")
    best_collision = float("inf")

    with tqdm(
        total=epochs * len(loader),
        desc="Training neural RQ",
        unit="batch",
        dynamic_ncols=True,
    ) as progress:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstructed, quant_loss, _ = model(
                    batch, use_sinkhorn=True
                )
                loss, _ = model.loss(reconstructed, quant_loss, batch)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite RQ training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += float(loss.detach())
                progress.update()
            epoch_loss = total_loss / max(1, len(loader))
            progress.set_postfix(
                epoch=epoch + 1,
                loss=f"{epoch_loss:.6f}",
            )

            should_eval = (
                (epoch + 1) % eval_every == 0 or epoch + 1 == epochs
            )
            if should_eval:
                codes = encode_neural_rq(
                    tensor.numpy(),
                    model,
                    device,
                    batch_size,
                    description=f"Evaluating epoch {epoch + 1}",
                    leave=False,
                )
                collision = 1.0 - len(set(map(tuple, codes))) / len(codes)
                key = (collision, epoch_loss)
                if key < best_key:
                    best_key = key
                    best_loss = epoch_loss
                    best_collision = collision
                    best_state = copy.deepcopy(model.state_dict())
                progress.write(
                    f"epoch={epoch + 1} loss={epoch_loss:.6f} "
                    f"collision_rate={collision:.6f}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)
    codes = encode_neural_rq(
        tensor.numpy(),
        model,
        device,
        batch_size,
        description="Encoding final SIDs",
    )
    return NeuralTrainResult(
        model=model,
        codes=codes,
        codebooks=model.codebooks(),
        best_loss=best_loss,
        collision_rate=best_collision,
    )


def encode_neural_rq(
    embeddings: np.ndarray,
    model,
    device: str,
    batch_size: int,
    description: str = "Encoding SIDs",
    leave: bool = True,
) -> np.ndarray:
    torch = _torch()
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(embeddings.astype(np.float32, copy=False))
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    chunks = []
    with torch.no_grad():
        for (batch,) in tqdm(
            loader,
            total=len(loader),
            desc=description,
            unit="batch",
            leave=leave,
            dynamic_ncols=True,
        ):
            chunks.append(model.get_indices(batch.to(device)).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.int32, copy=False)
