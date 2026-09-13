"""FAISS residual quantization backend adapted from MiniOneRec."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from .distance import prepare_numpy_values, validate_distance_metric


def _sizes_to_nbits(codebook_sizes: Sequence[int]) -> List[int]:
    sizes = [int(size) for size in codebook_sizes]
    if not sizes:
        raise ValueError("FAISS RQ-Kmeans requires at least one codebook")
    for size in sizes:
        if size <= 0 or size & (size - 1):
            raise ValueError(
                "FAISS codebook sizes must all be powers of two"
            )
    return [int(np.log2(size)) for size in sizes]


def _make_size_t_vector(faiss, values: Sequence[int]):
    for name in ("UInt64Vector", "Int64Vector", "Int32Vector"):
        vector_type = getattr(faiss, name, None)
        if vector_type is None:
            continue
        vector = vector_type()
        for value in values:
            vector.push_back(int(value))
        return vector
    return list(values)


def _build_quantizer(faiss, dimension: int, nbits: Sequence[int]):
    if len(set(nbits)) == 1:
        return faiss.ResidualQuantizer(dimension, len(nbits), nbits[0])
    vector = _make_size_t_vector(faiss, nbits)
    return faiss.ResidualQuantizer(dimension, vector)


def _unpack_codes(
    packed: np.ndarray,
    nbits: Sequence[int],
) -> np.ndarray:
    total_bytes = (sum(nbits) + 7) // 8
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1, total_bytes)
    codes = np.empty((len(packed), len(nbits)), dtype=np.int32)
    bit_offset = 0
    for level, level_nbits in enumerate(nbits):
        byte_offset = bit_offset // 8
        shift = bit_offset % 8
        bytes_needed = (shift + level_nbits + 7) // 8
        accumulator = np.zeros(len(packed), dtype=np.uint64)
        for byte_index in range(bytes_needed):
            accumulator |= (
                packed[:, byte_offset + byte_index].astype(np.uint64)
                << (8 * byte_index)
            )
        mask = (1 << level_nbits) - 1
        codes[:, level] = ((accumulator >> shift) & mask).astype(np.int32)
        bit_offset += level_nbits
    return codes


def _extract_codebooks(
    quantizer,
    codebook_sizes: Sequence[int],
) -> List[np.ndarray]:
    import faiss

    flat = faiss.vector_to_array(quantizer.codebooks).astype(np.float32)
    expected = sum(codebook_sizes) * quantizer.d
    if flat.size != expected:
        raise ValueError(
            f"FAISS returned {flat.size} codebook values; expected {expected}"
        )
    codebooks = []
    offset = 0
    for size in codebook_sizes:
        end = offset + int(size) * quantizer.d
        codebooks.append(flat[offset:end].reshape(int(size), quantizer.d).copy())
        offset = end
    return codebooks


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
    sizes = [int(size) for size in codebook_sizes]
    nbits = _sizes_to_nbits(sizes)
    values = np.ascontiguousarray(
        prepare_numpy_values(embeddings, metric, copy=False)
    )
    quantizer = _build_quantizer(faiss, values.shape[1], nbits)
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
        codes = _unpack_codes(packed, nbits)
        progress.update()

        progress.set_postfix_str("extracting codebooks")
        codebooks = _extract_codebooks(quantizer, sizes)
        progress.update()
    return codes, codebooks, quantizer
