"""Convert a Transformers PhoWhisper safetensors file to OpenAI Whisper format."""

import argparse
import json
import hashlib
import os
import re
import tempfile
from datetime import datetime, timezone
import urllib.request
from urllib.parse import quote
from pathlib import Path

import torch

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sidecar(path):
    return path.with_name(path.name + ".provenance.json")


def publish(temp, destination, overwrite=False):
    if overwrite:
        os.replace(temp, destination)
    else:
        os.link(temp, destination)  # Atomic create, never clobber a racing writer.
        temp.unlink()


def atomic_json(path, value, overwrite=False):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        publish(temp, path, overwrite)
    finally:
        temp.unlink(missing_ok=True)


def hub_metadata(model_id, revision):
    url = f"https://huggingface.co/api/models/{quote(model_id, safe='/')}/revision/{quote(revision, safe='')}"
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def weight_filename(files):
    for name in ("model.safetensors", "pytorch_model.bin"):
        if name in files:
            return name
    raise ValueError("No supported unsharded weights: supply a merged .bin or .safetensors and config locally; sharded checkpoints are unsupported")


def acquire(model_id, revision, root, overwrite=False):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id) or any(p in (".", "..") for p in model_id.split("/")):
        raise ValueError("Expected Hub model ID owner/name")
    metadata = hub_metadata(model_id, revision)
    sha = metadata["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Hub did not resolve an immutable commit SHA")
    filename = weight_filename([item["rfilename"] for item in metadata["siblings"]])
    directory = root / model_id / sha
    directory.mkdir(parents=True, exist_ok=True)
    source, config = directory / filename, directory / "config.json"
    record = directory / "download.provenance.json"
    identity = {"origin": "huggingface", "model_id": model_id, "resolved_revision": sha}
    if any(directory.iterdir()):
        try:
            cached = json.loads(record.read_text())
            valid = all(cached.get(k) == v for k, v in identity.items()) and cached["source_sha256"] == digest(source) and cached["config_sha256"] == digest(config)
        except (OSError, ValueError, KeyError):
            valid = False
        if valid:
            return source, config, {**cached, "requested_revision": revision}
        # Never repair a nonempty ambiguous cache, even with --overwrite.
        raise ValueError(f"Unverified or mismatching cache: {directory}; use a fresh --cache-dir")
    with tempfile.TemporaryDirectory(dir=directory) as staging:
        staging = Path(staging)
        for name in (filename, "config.json"):
            download(f"https://huggingface.co/{model_id}/resolve/{sha}/{name}", staging / name)
        provenance = {**identity, "requested_revision": revision, "source_sha256": digest(staging / filename), "config_sha256": digest(staging / "config.json")}
        created = []
        try:
            for name in (filename, "config.json"):
                publish(staging / name, directory / name)
                created.append(directory / name)
            atomic_json(record, provenance)
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
    return source, config, provenance


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def load_state_dict(source: Path) -> dict:
    if source.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(source, device="cpu")
    if source.suffix != ".bin":
        raise ValueError("Source must be an unsharded .bin or .safetensors")
    state = torch.load(source, map_location="cpu", weights_only=True)
    return state.get("state_dict", state)


def whisper_dimensions(config: dict) -> dict:
    return {
        "n_mels": config.get("num_mel_bins", 80),
        "n_audio_ctx": config["max_source_positions"],
        "n_audio_state": config["d_model"],
        "n_audio_head": config["encoder_attention_heads"],
        "n_audio_layer": config["encoder_layers"],
        "n_vocab": config["vocab_size"],
        "n_text_ctx": config["max_target_positions"],
        "n_text_state": config["d_model"],
        "n_text_head": config["decoder_attention_heads"],
        "n_text_layer": config["decoder_layers"],
    }


def convert_key(key: str) -> str | None:
    if key == "proj_out.weight":
        return None
    if key.startswith("model."):
        key = key.removeprefix("model.")

    key = key.replace("embed_tokens", "token_embedding")
    key = key.replace("embed_positions.weight", "positional_embedding")
    key = key.replace("layer_norm", "ln")
    key = key.replace("layers.", "blocks.")
    key = key.replace("self_attn.q_proj", "attn.query")
    key = key.replace("self_attn.k_proj", "attn.key")
    key = key.replace("self_attn.v_proj", "attn.value")
    key = key.replace("self_attn.out_proj", "attn.out")
    key = key.replace("self_attn_ln", "attn_ln")
    key = key.replace("encoder_attn.q_proj", "cross_attn.query")
    key = key.replace("encoder_attn.k_proj", "cross_attn.key")
    key = key.replace("encoder_attn.v_proj", "cross_attn.value")
    key = key.replace("encoder_attn.out_proj", "cross_attn.out")
    key = key.replace("encoder_attn_ln", "cross_attn_ln")
    key = key.replace("fc1", "mlp.0")
    key = key.replace("fc2", "mlp.2")
    key = key.replace("final_ln", "mlp_ln")
    key = key.replace("encoder.ln", "encoder.ln_post")
    return key


def validate_state(dimensions, state):
    from whisper.model import ModelDimensions, Whisper

    # Whisper's sparse alignment buffer does not support meta allocation.
    # Explicit CPU allocation avoids any GPU copy; assign reuses source tensors.
    with torch.device("cpu"):
        model = Whisper(ModelDimensions(**dimensions))
    model.load_state_dict(state, strict=True, assign=True)


def convert(source: Path, config_path: Path, output: Path, *, provenance=None, overwrite=False) -> None:
    metadata_path = sidecar(output)
    if not overwrite and (output.exists() or metadata_path.exists()):
        if provenance and output.is_file() and metadata_path.is_file():
            cached = json.loads(metadata_path.read_text())
            if (all(cached.get(k) == provenance.get(k) for k in ("origin", "model_id", "resolved_revision", "source_sha256", "config_sha256"))
                    and cached.get("native_sha256") == digest(output)
                    and cached.get("source_sha256") == digest(source)
                    and cached.get("config_sha256") == digest(config_path)):
                return
        raise FileExistsError(f"Refusing existing output/provenance: {output}")
    if output.resolve() in (source.resolve(), config_path.resolve()) or metadata_path.resolve() in (source.resolve(), config_path.resolve()):
        raise ValueError("Output must not replace input files")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_state = load_state_dict(source)
    model_state = {}
    for key, tensor in source_state.items():
        converted_key = convert_key(key)
        if converted_key is not None:
            if converted_key in model_state:
                raise ValueError(f"Duplicate converted key: {converted_key}")
            model_state[converted_key] = tensor

    dimensions = whisper_dimensions(config)
    validate_state(dimensions, model_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        torch.save({"dims": dimensions, "model_state_dict": model_state}, temp)
        record = {**(provenance or {"origin": "local/unverified"}),
                  "source_sha256": digest(source), "config_sha256": digest(config_path),
                  "native_sha256": digest(temp), "dimensions": dimensions,
                  "timestamp": datetime.now(timezone.utc).isoformat()}
        publish(temp, output, overwrite)
        try:
            atomic_json(metadata_path, record, overwrite)
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    finally:
        temp.unlink(missing_ok=True)
    print(f"Saved PyTorch checkpoint to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-id", default="vinai/PhoWhisper-small")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/official"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    provenance = None
    if args.download:
        if args.source or args.config:
            parser.error("--download cannot be combined with local --source/--config")
        args.source, args.config, provenance = acquire(args.model_id, args.revision, args.cache_dir)
        args.output = args.output or args.source.parent / "native.pt"
    elif not (args.source and args.config and args.output):
        parser.error("Local conversion requires explicit --source, --config and --output; model ID does not authenticate local files")
    convert(args.source, args.config, args.output, provenance=provenance, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
