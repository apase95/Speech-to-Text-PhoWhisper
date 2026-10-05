import argparse
import csv
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf

DATASET = "tuandaodev/viet-bud500-formatted"


def numeric_audio_id(path: str) -> int:
    name = Path(path).stem
    if not name.startswith("audio") or not name[5:].isdigit():
        raise ValueError(f"Unexpected audio path: {path}")
    return int(name[5:])


def read_metadata(paths: list[Path]) -> dict[int, tuple[str, str]]:
    samples = {}
    for path in paths:
        split = path.stem.removeprefix("metadata_")
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="|"):
                samples[numeric_audio_id(row["audio_file"])] = (row["text"], split)
    return samples


def download_one(kaggle: Path, audio_id: int, audio_dir: Path) -> Path:
    destination = audio_dir / f"audio{audio_id}.wav"
    if destination.is_file():
        sf.info(destination)
        return destination
    subprocess.run(
        [
            str(kaggle), "datasets", "download", DATASET,
            "--file", f"wavs/audio{audio_id}.wav",
            "--path", str(audio_dir), "--unzip", "--quiet",
        ],
        check=True,
    )
    sf.info(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a deterministic Viet-BUD500 subset")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmark/viet-bud500"))
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/manifests/viet_bud500_1000.csv"))
    args = parser.parse_args()

    metadata_paths = [args.data_dir / "metadata_train.csv", args.data_dir / "metadata_test.csv"]
    missing_metadata = [str(path) for path in metadata_paths if not path.is_file()]
    if missing_metadata:
        parser.error(f"Missing metadata files: {', '.join(missing_metadata)}")

    metadata = read_metadata(metadata_paths)
    ids = list(range(args.start_id, args.start_id + args.count))
    missing_ids = [audio_id for audio_id in ids if audio_id not in metadata]
    if missing_ids:
        parser.error(f"Metadata is missing IDs, first: {missing_ids[:10]}")

    audio_dir = args.data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    kaggle = Path(__file__).parents[1] / ".venv/bin/kaggle"
    if not kaggle.is_file():
        parser.error(f"Kaggle CLI not found: {kaggle}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, kaggle, audio_id, audio_dir): audio_id for audio_id in ids}
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed % 50 == 0 or completed == len(ids):
                print(f"Downloaded/verified {completed}/{len(ids)}", flush=True)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "audio_path", "reference", "split", "sha256"])
        writer.writeheader()
        for audio_id in ids:
            path = audio_dir / f"audio{audio_id}.wav"
            writer.writerow({
                "id": f"audio{audio_id}",
                "audio_path": path.as_posix(),
                "reference": metadata[audio_id][0],
                "split": metadata[audio_id][1],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
