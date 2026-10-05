"""Convert a Transformers PhoWhisper safetensors file to OpenAI Whisper format."""

import argparse
import json
import urllib.request
from pathlib import Path

import torch

MODEL_URL = "https://huggingface.co/vinai/PhoWhisper-base/resolve/main/pytorch_model.bin"
CONFIG_URL = "https://huggingface.co/vinai/PhoWhisper-base/resolve/main/config.json"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def load_state_dict(source: Path) -> dict:
    if source.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(source, device="cpu")
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


def convert(source: Path, config_path: Path, output: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_state = load_state_dict(source)
    model_state = {}
    for key, tensor in source_state.items():
        converted_key = convert_key(key)
        if converted_key is not None:
            model_state[converted_key] = tensor

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"dims": whisper_dimensions(config), "model_state_dict": model_state}, output
    )
    print(f"Saved PyTorch checkpoint to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("models/pytorch_model.bin"))
    parser.add_argument("--config", type=Path, default=Path("models/config.json"))
    parser.add_argument("--output", type=Path, default=Path("models/phowhisper-base.pt"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if args.download:
        if not args.source.exists():
            download(MODEL_URL, args.source)
        if not args.config.exists():
            download(CONFIG_URL, args.config)
    if not args.source.is_file() or not args.config.is_file():
        parser.error("Thiếu source/config; dùng --download hoặc truyền đường dẫn local")
    convert(args.source, args.config, args.output)


if __name__ == "__main__":
    main()
