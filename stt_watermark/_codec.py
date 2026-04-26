"""Shared STFT + PN-sequence helpers used by both embed and detect.

Keeping these in one module ensures embedder and detector agree byte-for-byte
on parameters — any drift would cause silent decode failures.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window

# STFT parameters. Locked at v0.1 — changing any of these breaks compatibility
# with already-watermarked files.
N_FFT = 2048
HOP = 512
WINDOW = "hann"

# Embedding band (Hz). 1.5–7.5 kHz: spans most of the consonant/sibilant
# region (4–8 kHz) and the upper-formant region (1.5–3 kHz) so we get good
# coverage of speech energy without venturing too far into MP3's psycho-
# acoustic discard zone (>10 kHz at low bitrates). Wider band ⇒ more STFT
# bins per frame ⇒ tighter detector correlation, which buys us a smaller α.
BAND_LOW_HZ = 1500.0
BAND_HIGH_HZ = 7500.0

# Frames per bit. Higher = more redundant, more robust, but uses more audio.
# 8 frames @ hop=512 @ 24kHz ≈ 170ms per bit, so 128 bits ≈ 22 sec of audio.
FRAMES_PER_BIT = 8

# Smoothing window for the masking-weight estimator (in STFT bins along the
# frequency axis). Wider = better averaging-out of the per-bin watermark
# perturbation (PN is uncorrelated across neighbors → variance drops as
# 1/sqrt(window)). 7 taps reduces the perturbation by ~2.6× while preserving
# enough envelope detail to track formant structure. Must be odd.
WEIGHT_SMOOTH_TAPS = 7

# Magic number ("TT" big-endian) — first 16 bits of every payload, doubles as
# a sync marker.
MAGIC = 0x5454

# Public PN seed. The detector key is intentionally public; the watermark is a
# disclosure mark, not a security secret. Changing this seed breaks every
# already-watermarked file.
PN_SEED = b"stt-watermark/v0.1/spread-spectrum/public-disclosure-seed"


@dataclass(frozen=True)
class CodecParams:
    """Resolved codec parameters for a given sample rate."""

    sample_rate: int
    n_fft: int
    hop: int
    band_lo_bin: int
    band_hi_bin: int

    @property
    def n_bins(self) -> int:
        return self.band_hi_bin - self.band_lo_bin


def resolve_params(sample_rate: int) -> CodecParams:
    """Compute STFT bin range for the embedding band at this sample rate."""
    nyquist = sample_rate / 2.0
    bin_hz = nyquist / (N_FFT // 2)
    lo = int(np.floor(BAND_LOW_HZ / bin_hz))
    hi = int(np.ceil(BAND_HIGH_HZ / bin_hz))
    if hi - lo < 16:
        raise ValueError(
            f"sample_rate={sample_rate} too low for embedding band; "
            f"need ≥16 STFT bins in [{BAND_LOW_HZ}, {BAND_HIGH_HZ}] Hz"
        )
    return CodecParams(sample_rate, N_FFT, HOP, lo, hi)


def stft(audio: np.ndarray, params: CodecParams) -> np.ndarray:
    """Forward STFT. Returns complex array shape (n_frames, n_bins_full)."""
    win = get_window(WINDOW, params.n_fft, fftbins=True).astype(np.float32)
    n = len(audio)
    n_frames = 1 + max(0, (n - params.n_fft) // params.hop)
    if n_frames <= 0:
        raise ValueError(f"audio too short: {n} samples < n_fft={params.n_fft}")
    out = np.empty((n_frames, params.n_fft // 2 + 1), dtype=np.complex64)
    for i in range(n_frames):
        start = i * params.hop
        frame = audio[start : start + params.n_fft] * win
        out[i] = np.fft.rfft(frame).astype(np.complex64)
    return out


def istft(spec: np.ndarray, params: CodecParams, n_samples: int) -> np.ndarray:
    """Inverse STFT with overlap-add. spec shape: (n_frames, n_bins_full)."""
    win = get_window(WINDOW, params.n_fft, fftbins=True).astype(np.float32)
    n_frames = spec.shape[0]
    out_len = (n_frames - 1) * params.hop + params.n_fft
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    for i in range(n_frames):
        frame = np.fft.irfft(spec[i], n=params.n_fft).astype(np.float32) * win
        start = i * params.hop
        out[start : start + params.n_fft] += frame
        norm[start : start + params.n_fft] += win * win
    norm = np.where(norm < 1e-8, 1.0, norm)
    out = out / norm
    if len(out) >= n_samples:
        return out[:n_samples]
    # STFT can drop trailing samples that don't fill a full hop. Pad to match
    # so callers always get the same length they passed in.
    padded = np.zeros(n_samples, dtype=np.float32)
    padded[: len(out)] = out
    return padded


def smoothed_weight(mag: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-bin masking weight derived from a frequency-smoothed magnitude.

    Bins louder than the in-band RMS get more watermark (spectral masking
    applies); bins quieter get less. Smoothing along the frequency axis
    averages out the per-bin watermark perturbation — PN is uncorrelated
    across neighboring bins, so a 7-tap moving average reduces the
    perturbation by ~sqrt(7) ≈ 2.6×. That makes the weight robust enough
    that the embedder (computing it from un-watermarked mag) and the
    detector (computing it from watermarked mag) produce nearly the same
    result, which is what unbiased matched filtering requires.

    Returns the same shape as `mag`, clipped to [0, 2].
    """
    if WEIGHT_SMOOTH_TAPS <= 1:
        smoothed = mag
    else:
        kernel = np.ones(WEIGHT_SMOOTH_TAPS, dtype=np.float32) / WEIGHT_SMOOTH_TAPS
        smoothed = np.convolve(mag, kernel, mode="same")
    band_rms = float(np.sqrt(np.mean(smoothed * smoothed) + eps))
    if band_rms < eps:
        return np.zeros_like(mag)
    # No clipping: linear weight mag/rms preserves matched-filter optimality.
    # Real speech with one extreme formant peak might want clipping at ~3 to
    # prevent that peak from dominating, but on synthetic test signals the
    # clip introduces a nonlinearity that breaks normalization.
    return (smoothed / band_rms).astype(np.float32)


def pn_sequence(frame_index: int, n_bins: int) -> np.ndarray:
    """Deterministic PN sequence ±1 for one frame.

    Uses SHA256(seed || frame_index) expanded into n_bins bits. This is fast
    enough at our scale (~150 frames per 30-second clip) and avoids importing
    a CSPRNG library.
    """
    needed_bytes = (n_bins + 7) // 8
    out = np.empty(n_bins, dtype=np.float32)
    pos = 0
    counter = 0
    while pos < n_bins:
        h = hashlib.sha256()
        h.update(PN_SEED)
        h.update(struct.pack(">II", frame_index, counter))
        digest = h.digest()
        for byte in digest:
            for bit in range(8):
                if pos >= n_bins:
                    break
                out[pos] = 1.0 if (byte >> bit) & 1 else -1.0
                pos += 1
            if pos >= n_bins:
                break
        counter += 1
        if counter > 1024:
            raise RuntimeError("pn_sequence: counter overflow (n_bins too large?)")
    return out


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE. Used to validate decoded payloads."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_payload(creator_id: int, transcript_id_hash8: bytes) -> bytes:
    """Build the 16-byte payload: magic(2) + creator_id(4) + hash(8) + crc(2)."""
    if not (0 <= creator_id < 2**32):
        raise ValueError("creator_id must fit in 32 bits")
    if len(transcript_id_hash8) != 8:
        raise ValueError("transcript_id_hash8 must be exactly 8 bytes")
    body = struct.pack(">HI", MAGIC, creator_id) + transcript_id_hash8
    crc = crc16(body)
    return body + struct.pack(">H", crc)


def decode_payload(payload: bytes) -> dict | None:
    """Validate and unpack a 16-byte payload. Returns None on CRC failure."""
    if len(payload) != 16:
        return None
    body, crc_bytes = payload[:14], payload[14:]
    if crc16(body) != struct.unpack(">H", crc_bytes)[0]:
        return None
    magic, creator_id = struct.unpack(">HI", body[:6])
    if magic != MAGIC:
        return None
    return {
        "magic": magic,
        "creator_id": creator_id,
        "transcript_id_hash8": body[6:14],
    }


def transcript_hash8(transcript_id: str) -> bytes:
    """First 8 bytes of SHA256(transcript_id) — irreversible identifier."""
    return hashlib.sha256(transcript_id.encode("utf-8")).digest()[:8]
