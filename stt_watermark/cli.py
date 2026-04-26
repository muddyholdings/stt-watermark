"""Command-line interface for stt-watermark."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import soundfile as sf

from stt_watermark import __version__
from stt_watermark.detect import detect
from stt_watermark.embed import embed


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return audio, int(sr)


def _cmd_embed(args: argparse.Namespace) -> int:
    audio, sr = _load_mono(args.input)
    watermarked, info = embed(
        audio,
        sample_rate=sr,
        creator_id=args.creator_id,
        transcript_id=args.transcript_id,
        strength=args.strength,
    )
    sf.write(args.output, watermarked, sr)
    if args.json:
        print(json.dumps(info))
    else:
        print(
            f"embedded: {info['n_bits']} bits in {info['n_frames']} frames @ "
            f"{info['sample_rate']}Hz, strength={info['strength_used']}, "
            f"payload={info['payload_hex']}"
        )
        print(f"wrote {args.output}")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    audio, sr = _load_mono(args.input)
    result = detect(audio, sample_rate=sr)
    if args.json:
        print(
            json.dumps(
                {
                    "found": result.found,
                    "payload_hex": result.payload.hex() if result.payload else None,
                    "creator_id": result.creator_id,
                    "transcript_id_hash8_hex": (
                        result.transcript_id_hash8.hex()
                        if result.transcript_id_hash8
                        else None
                    ),
                    "confidence": result.confidence,
                }
            )
        )
    else:
        if result.found:
            print(f"WATERMARK FOUND  confidence={result.confidence:.3f}")
            print(f"  payload:        {result.payload.hex()}")
            print(f"  creator_id:     {result.creator_id}")
            print(f"  transcript_id:  {result.transcript_id_hash8.hex()} (sha256[:8])")
        else:
            print(f"no watermark detected  confidence={result.confidence:.3f}")
    return 0 if result.found or args.allow_missing else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stt-watermark",
        description="Embed/detect inaudible spread-spectrum watermarks in audio.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_embed = sub.add_parser("embed", help="Embed a watermark into a WAV file.")
    p_embed.add_argument("input", help="Input audio (WAV/FLAC/OGG, mono or stereo).")
    p_embed.add_argument("output", help="Output WAV path.")
    p_embed.add_argument("--creator-id", type=int, default=0, help="32-bit creator id.")
    p_embed.add_argument(
        "--transcript-id",
        type=str,
        default="",
        help="Arbitrary string hashed (SHA256[:8]) into the payload.",
    )
    p_embed.add_argument(
        "--strength", type=float, default=0.05, help="Embedding gain α (default 0.05)."
    )
    p_embed.add_argument("--json", action="store_true", help="Emit info as JSON.")
    p_embed.set_defaults(func=_cmd_embed)

    p_detect = sub.add_parser("detect", help="Detect a watermark in an audio file.")
    p_detect.add_argument("input", help="Input audio.")
    p_detect.add_argument("--json", action="store_true", help="Emit result as JSON.")
    p_detect.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit 0 even if no watermark is found (default: exit 1 on miss).",
    )
    p_detect.set_defaults(func=_cmd_detect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
