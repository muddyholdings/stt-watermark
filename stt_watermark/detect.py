"""Spread-spectrum watermark detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from stt_watermark._codec import (
    FRAMES_PER_BIT,
    CodecParams,
    decode_payload,
    pn_sequence,
    resolve_params,
    stft,
)


@dataclass(frozen=True)
class DetectResult:
    """Result of a detection attempt."""

    found: bool
    payload: Optional[bytes]
    confidence: float  # 0.0–1.0; ratio of bits whose soft margin exceeds threshold
    creator_id: Optional[int] = None
    transcript_id_hash8: Optional[bytes] = None

    def __bool__(self) -> bool:
        return self.found


def detect(
    audio: np.ndarray,
    sample_rate: int,
    sync_search_frames: int = 8,
) -> DetectResult:
    """Detect a 128-bit watermark in mono audio.

    Args:
        audio: 1-D numpy float32 mono.
        sample_rate: 16000, 24000, 44100, 48000.
        sync_search_frames: Slide the bit grid by ±this many frames to
            recover from small speed/timing offsets.

    Returns DetectResult. .found is True iff payload CRC validates.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio must be 1-D mono")

    params = resolve_params(sample_rate)
    spec = stft(audio, params)
    n_bits = 128
    frames_needed = n_bits * FRAMES_PER_BIT
    if spec.shape[0] < frames_needed:
        return DetectResult(found=False, payload=None, confidence=0.0)

    best: Optional[DetectResult] = None
    for offset in _offset_candidates(sync_search_frames):
        if offset < 0:
            continue
        if offset + frames_needed > spec.shape[0]:
            continue
        bits, soft_margins = _decode_bits(spec, params, offset, n_bits)
        payload = _bits_to_bytes(bits)
        decoded = decode_payload(payload)
        if decoded is not None:
            confidence = float(np.mean(soft_margins > 0.1))
            return DetectResult(
                found=True,
                payload=payload,
                confidence=confidence,
                creator_id=decoded["creator_id"],
                transcript_id_hash8=decoded["transcript_id_hash8"],
            )
        if best is None:
            best = DetectResult(
                found=False,
                payload=None,
                confidence=float(np.mean(np.abs(soft_margins))),
            )

    return best or DetectResult(found=False, payload=None, confidence=0.0)


def _offset_candidates(span: int):
    """Yield frame offsets in widening order: 0, +1, -1, +2, -2, ..."""
    yield 0
    for d in range(1, span + 1):
        yield d
        yield -d


def _decode_bits(
    spec: np.ndarray, params: CodecParams, frame_offset: int, n_bits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Flat-weighted correlation against PN(frame_idx).

    A naive matched filter that re-derives the embedder's per-bin weight
    from the received audio looks attractive on paper, but the watermark
    perturbation is itself correlated with the weight, so recomputed-weight
    correlation has a sign-dependent bias term that pollutes detection. The
    flat-weighted correlator is suboptimal (loses ~3 dB vs an oracle weight
    detector) but unbiased — it just sees less embedding energy in the
    quiet bins, which the embedder skipped on purpose anyway.
    Returns (hard_bits, soft_margins).
    """
    lo, hi = params.band_lo_bin, params.band_hi_bin
    eps = 1e-8
    bits = np.zeros(n_bits, dtype=np.int8)
    margins = np.zeros(n_bits, dtype=np.float32)
    for bit_idx in range(n_bits):
        accum = 0.0
        for rep in range(FRAMES_PER_BIT):
            frame_idx = frame_offset + bit_idx * FRAMES_PER_BIT + rep
            if frame_idx >= spec.shape[0] or frame_idx < 0:
                continue
            pn = pn_sequence(bit_idx * FRAMES_PER_BIT + rep, params.n_bins)
            log_mag = np.log(np.abs(spec[frame_idx, lo:hi]) + eps)
            centered = log_mag - float(np.mean(log_mag))
            corr = float(np.dot(centered, pn) / params.n_bins)
            accum += corr
        margins[bit_idx] = accum / FRAMES_PER_BIT
        bits[bit_idx] = 1 if accum > 0 else 0
    return bits, margins


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    """128 bits MSB-first → 16 bytes."""
    out = bytearray(len(bits) // 8)
    for i in range(len(bits) // 8):
        b = 0
        for j in range(8):
            b = (b << 1) | int(bits[i * 8 + j])
        out[i] = b
    return bytes(out)
