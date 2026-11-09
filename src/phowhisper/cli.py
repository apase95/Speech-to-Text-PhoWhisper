"""Lightweight console entry point; --help never imports model dependencies."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='PhoWhisper + GPT-2: Vietnamese Speech-to-Text')
    parser.add_argument('--audio', required=True, help='Path to input audio file')
    parser.add_argument('--time-stretch-rate', type=float, default=1.0)
    parser.add_argument('--rescore', action='store_true')
    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--confidence-threshold', type=float, default=-0.020)
    args = parser.parse_args()
    path = Path(args.audio)
    if not path.exists():
        raise FileNotFoundError(f'Không tìm thấy file: {path}')
    from phowhisper.pipeline import transcribe
    print('Transcript:')
    print(transcribe(str(path), time_stretch_rate=args.time_stretch_rate,
        rescore=args.rescore, beam_size=args.beam_size,
        confidence_threshold=None if args.confidence_threshold < -100 else args.confidence_threshold))


if __name__ == '__main__':
    main()
