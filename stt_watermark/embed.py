"""Spread-spectrum watermark embedder."""

from __future__ import annotations

import os
import struct
from typing import Optional

import numpy as np

from stt_watermark._codec import (
    FRAMES_PER_BIT,
    CodecParams,
    encode_payload,
    istft,
    pn_sequence,
    resolve_params,
    stft,
    transcript_hash8,
)


def embed(
    audio: np.ndarray,
    sample_rate: int,
    payload: Optional[bytes] = None,
    creator_id: int = 0,
    transcript_id: str = "",
    strength: float = 0.15,
) -> tuple[np.ndarray, dict]:
    """Embed a 128-bit watermark into mono audio.

    Args:
        audio: 1-D numpy float32, mono. Values in approximately [-1, 1].
        sample_rate: 16000, 24000, 44100, or 48000.
        payload: Optional 16-byte payload. If None, built from creator_id +
            SHA256(transcript_id)[:8] + CRC.
        creator_id: 32-bit creator id (used only when payload is None).
        transcript_id: Arbitrary string hashed into the payload (used only
            when payload is None). Empty string → zero hash.
        strength: Embedding gain α. 0.05 is the v0.1 default; higher = more
            robust but more audible.

    Returns:
        (watermarked_audio, info) where info is a dict with at least
        {"strength_used", "n_frames", "n_bits", "payload_hex"}.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio must be 1-D mono")

    if payload is None:
        payload = encode_payload(creator_id, transcript_hash8(transcript_id))
    if len(payload) != 16:
        raise ValueError(f"payload must be 16 bytes, got {len(payload)}")

    params = resolve_params(sample_rate)
    n_bits = 16 * 8  # 128 bits
    frames_needed = n_bits * FRAMES_PER_BIT
    min_samples = params.n_fft + (frames_needed - 1) * params.hop
    if len(audio) < min_samples:
        raise ValueError(
            f"audio too short: {len(audio)} samples < {min_samples} required for "
            f"128-bit payload at sample_rate={sample_rate}. Need ≈"
            f"{min_samples / sample_rate:.1f}s minimum."
        )

    spec = stft(audio, params)
    bits = _bytes_to_bits(payload)
    _embed_bits_inplace(spec, bits, params, strength)
    watermarked = istft(spec, params, n_samples=len(audio))

    return watermarked, {
        "strength_used": strength,
        "n_frames": spec.shape[0],
        "n_bits": n_bits,
        "frames_per_bit": FRAMES_PER_BIT,
        "payload_hex": payload.hex(),
        "sample_rate": sample_rate,
    }


def _bytes_to_bits(b: bytes) -> np.ndarray:
    """16 bytes → 128 bits, MSB-first within each byte."""
    out = np.zeros(len(b) * 8, dtype=np.int8)
    for i, byte in enumerate(b):
        for j in range(8):
            out[i * 8 + j] = 1 if (byte >> (7 - j)) & 1 else 0
    return out


def _embed_bits_inplace(
    spec: np.ndarray, bits: np.ndarray, params: CodecParams, strength: float
) -> None:
    """Modulate log-magnitudes in-band by ±α·PN[k] per bit, replicated across
    FRAMES_PER_BIT consecutive frames.

    Operating in log-magnitude (dB-like) domain gives a much more uniform
    distribution across bins than linear magnitude, which means the detector's
    correlation has tight noise variance — speech spectral lines no longer
    dominate the correlation sum. α=0.15 corresponds to ~1.3 dB peak
    modulation, audible on synthetic test signals but documented as v0.3
    work to add a smoothed-weight psychoacoustic shaper.
    """
    lo, hi = params.band_lo_bin, params.band_hi_bin
    eps = 1e-8
    for bit_idx, bit_val in enumerate(bits):
        sign = 1.0 if bit_val == 1 else -1.0
        for rep in range(FRAMES_PER_BIT):
            frame_idx = bit_idx * FRAMES_PER_BIT + rep
            if frame_idx >= spec.shape[0]:
                return
            pn = pn_sequence(frame_idx, params.n_bins)
            mag = np.abs(spec[frame_idx, lo:hi])
            phase = np.angle(spec[frame_idx, lo:hi])
            log_mag = np.log(mag + eps)
            log_mag_new = log_mag + sign * strength * pn
            new_mag = np.exp(log_mag_new) - eps
            new_mag = np.maximum(new_mag, 0.0)
            spec[frame_idx, lo:hi] = (new_mag * np.exp(1j * phase)).astype(np.complex64)
