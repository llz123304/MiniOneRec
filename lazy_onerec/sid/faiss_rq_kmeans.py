"""FAISS residual quantization backend adapted from MiniOneRec."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from .distance import prepare_numpy_values, validate_distance_metric


def _validate_sizes(codebook_sizes: Sequence[int]) -> int:
    sizes = [int(size) for size in codebook_sizes]
    if len(set(sizes)) != 1:
        raise ValueError("FAISS RQ-Kmeans requires equal codebook sizes")
    size = sizes[0]
    if size <= 0 or size & (size - 1):
        raise ValueError("FAISS codebook size must be a power of two")
    return size


def _first_nbits(quantizer) -> int:
    import faiss

    if isinstance(quantizer.nbits, int):
        return quantizer.nbits
    return int(faiss.vector_to_array(quantizer.nbits).ravel()[0])


def _unpack_codes(
    packed: np.ndarray, nbits: int, n_levels: int
) -> np.ndarray:
    packed_ints = np.zeros(packed.shape[0], dtype=np.int64)
    for byte in range(packed.shape[1]):
        packed_ints |= packed[:, byte].astype(np.int64) << (8 * byte)
    mask = (1 << nbits) - 1
    codes = np.empty((packed.shape[0], n_levels), dtype=np.int32)
    for level in range(n_levels):
        codes[:, level] = (packed_ints >> (level * nbits)) & mask
    return codes


def _extract_codebooks(quantizer) -> List[np.ndarray]:
    import faiss

    size = 1 << _first_nbits(quantizer)
    flat = faiss.vector_to_array(quantizer.codebooks).astype(np.float32)
    stacked = flat.reshape(quantizer.M, size, quantizer.d)
    return [stacked[level].copy() for level in range(quantizer.M)]


def faiss_residual_kmeans(
    embeddings: np.ndarray,
    codebook_sizes: Sequence[int],
    beam_size: int = 1,
    distance_metric: str = "euclidean",
) -> Tuple[np.ndarray, List[np.ndarray], object]:
    """Return raw codes, codebooks, and the trained FAISS quantizer."""
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "Install faiss-cpu or faiss-gpu to use RQ-Kmeans"
        ) from exc

    metric = validate_distance_metric(distance_metric)
    size = _validate_sizes(codebook_sizes)
    values = np.ascontiguousarray(
        prepare_numpy_values(embeddings, metric, copy=False)
    )
    nbits = int(np.log2(size))
    quantizer = faiss.ResidualQuantizer(
        values.shape[1], len(codebook_sizes), nbits
    )
    quantizer.train_type = faiss.ResidualQuantizer.Train_default
    quantizer.max_beam_size = int(beam_size)
    with tqdm(
        total=3,
        desc="FAISS RQ-Kmeans",
        unit="stage",
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix_str("training codebooks")
        quantizer.train(values)
        progress.update()

        progress.set_postfix_str("encoding items")
        packed = quantizer.compute_codes(values)
        if packed.ndim == 1:
            n_bytes = (len(codebook_sizes) * nbits + 7) // 8
            packed = packed.reshape(-1, n_bytes)
        if nbits == 8:
            codes = packed[:, : len(codebook_sizes)].astype(np.int32)
        else:
            codes = _unpack_codes(packed, nbits, len(codebook_sizes))
        progress.update()

        progress.set_postfix_str("extracting codebooks")
        codebooks = _extract_codebooks(quantizer)
        progress.update()
    return codes, codebooks, quantizer
