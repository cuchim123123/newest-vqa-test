"""Standalone inference script for the VQA system."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from PIL import Image

from src.config import Config
from src.models.vqa_model import VQAModel
from src.utils.helpers import get_device, decode_sequence


class VQAInferencePipeline:
    """Load a trained VQA checkpoint and run inference on arbitrary images."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/default.yaml",
        device_str: str = "auto",
    ):
        self.device = get_device() if device_str == "auto" else torch.device(device_str)
        cfg = Config.from_yaml(config_path)

        # Load checkpoint (may be single-model or deploy-all artifact)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Determine checkpoint format
        if "q_vocab" in ckpt:
            # Deploy artifact produced by VQA.ipynb Cell 12
            self.q_vocab = ckpt["q_vocab"]
            self.a_vocab = ckpt["a_vocab"]
            model_states: dict = ckpt["model_states"]
            model_cfg = ckpt.get("config", {})
        elif "model_state_dict" in ckpt:
            # Single-model checkpoint from trainer
            vocab_path = os.path.join(os.path.dirname(checkpoint_path), "..", "data", "processed", "vocab.pth")
            if not os.path.exists(vocab_path):
                vocab_path = "data/processed/vocab.pth"
            vocabs = torch.load(vocab_path, map_location=self.device, weights_only=False)
            self.q_vocab = vocabs["q_vocab"]
            self.a_vocab = vocabs["a_vocab"]
            model_states = {"model": ckpt["model_state_dict"]}
            model_cfg = {}
        else:
            raise ValueError(f"Unrecognised checkpoint format. Keys: {list(ckpt.keys())}")

        # Build models
        self.models: dict[str, VQAModel] = {}
        for name, state_dict in model_states.items():
            variant_cfg = cfg.model_variants.get(name, {})
            model = VQAModel(
                q_vocab_size=len(self.q_vocab),
                a_vocab_size=len(self.a_vocab),
                embed_size=model_cfg.get("embed_size", cfg.model.embed_size),
                hidden_size=model_cfg.get("hidden_size", cfg.model.hidden_size),
                num_layers=model_cfg.get("num_layers", cfg.model.num_layers),
                dropout=model_cfg.get("dropout", cfg.model.dropout),
                bidirectional=model_cfg.get("bidirectional", cfg.model.bidirectional),
                **variant_cfg,
            )
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self.models[name] = model

        # Image transform (same as validation)
        img_size = model_cfg.get("image_size", cfg.data.image_size)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ── public API ──────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        model_name: str | None = None,
        beam_width: int = 5,
    ) -> dict[str, str]:
        """Return {model_name: answer_string} for one or all loaded models."""
        img_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        tokens = [2] + self.q_vocab.numericalize(question) + [3]  # SOS + tokens + EOS
        q_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        q_length = torch.tensor([len(tokens)], dtype=torch.long).to(self.device)

        targets = {model_name: self.models[model_name]} if model_name else self.models
        results: dict[str, str] = {}
        for mname, model in targets.items():
            gen_ids = model.generate(img_tensor, q_tensor, q_length, use_beam=True, beam_width=beam_width)
            answer = decode_sequence(gen_ids.squeeze(0).tolist(), self.a_vocab)
            results[mname] = answer
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="VQA Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--question", type=str, required=True, help="Question text")
    parser.add_argument("--checkpoint", type=str, default="vqa_deploy_all_models.pth",
                        help="Path to checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model variant (e.g. M4_Pretrained_Attn). Omit to run all.")
    parser.add_argument("--beam", type=int, default=5, help="Beam width for generation")
    args = parser.parse_args()

    pipe = VQAInferencePipeline(args.checkpoint, args.config)
    answers = pipe.predict(Image.open(args.image), args.question,
                           model_name=args.model, beam_width=args.beam)

    print(f"\nQuestion: {args.question}")
    for name, ans in answers.items():
        print(f"  [{name}] → {ans}")


if __name__ == "__main__":
    main()