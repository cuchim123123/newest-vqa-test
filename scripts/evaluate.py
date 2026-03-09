import argparse
import os
import sys
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset

# Add root directory to sys.path to recognize src package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.dataset import AOKVQA_Dataset, collate_fn
from src.models.vqa_model import VQAModel
from src.engine.evaluator import evaluate_model, evaluate_by_question_type, get_failure_cases
from src.utils.helpers import get_device, setup_logging
from src.utils.visualization import plot_radar_chart, plot_bar_chart

def main() -> None:
    # 1. Initialize parameters and environment
    parser = argparse.ArgumentParser(description="Evaluate VQA models")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--models", nargs="+", help="Specific models to evaluate (e.g., M4_Pretrained_Attn)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = get_device()
    logger = setup_logging(cfg.log_dir)
    logger.info(f"Start evaluating models on device: {device}")

    # 2. Load Vocabulary saved from training step
    vocab_path = "data/processed/vocab.pth"
    if not os.path.exists(vocab_path):
        logger.error(f"File not found from vocab at {vocab_path}. Run train.py first!")
        sys.exit(1)
        
    vocabs = torch.load(vocab_path, map_location=device, weights_only=False)
    question_vocab = vocabs["q_vocab"]
    answer_vocab = vocabs["a_vocab"]
    logger.info(f"Loaded vocab: {len(question_vocab)} from (question) and {len(answer_vocab)} from (answer).")

    # 3. Load Dataset and prepare DataLoader (Validation set only)
    logger.info("Loading validation dataset from HuggingFace...")
    hf_val = load_dataset(cfg.data.hf_id, split="validation")
    val_data = list(hf_val)

    # Normalize images identical to training
    transform = transforms.Compose([
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_ds = AOKVQA_Dataset(val_data, question_vocab, answer_vocab, transform)
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory
    )

    all_results = {}
    variants_to_eval = args.models or list(cfg.model_variants.keys())
    
    # 4. Evaluating from variants of architecture
    for name in variants_to_eval:
        logger.info(f"\n{'='*50}\nEvaluating architecture: {name}\n{'='*50}")
        
        # Initialize model based on configuration (With/Without Attention, Pretrained/Scratch)
        variant_cfg = cfg.model_variants.get(name, {})
        model = VQAModel(
            q_vocab_size=len(question_vocab),
            a_vocab_size=len(answer_vocab),
            embed_size=cfg.model.embed_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            bidirectional=cfg.model.bidirectional,
            num_answers=cfg.model.num_answers,
            **variant_cfg
        ).to(device)
        
        # Evaluate using Beam Search for smoothest sequential generation
        res = evaluate_model(
            model=model, 
            test_loader=val_loader, 
            answer_vocab=answer_vocab, 
            question_vocab=question_vocab, 
            device=device, 
            ckpt_dir=cfg.ckpt_dir,
            name=name,
            beam_width=cfg.train.beam_width
        )
        
        all_results[name] = res["metrics"]
        
        # Deep analysis: Performance by question type (What, Where, Why, Yes/No...)
        q_results = evaluate_by_question_type(res["preds"], res["refs"], res["questions"])
        
        # Extract top 20 worst failure cases based on low F1-score
        failures = get_failure_cases(res["preds"], res["refs"], res["questions"], n=20)
        
        # Save failure cases to JSON file
        fail_path = os.path.join(cfg.log_dir, f"failures_{name}.json")
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=4, ensure_ascii=False)
        logger.info(f"saved failure cases to: {fail_path}")

    # 5. Aggregate and visualize results
    logger.info("\nCreating comparison charts between models...")
    radar_path = os.path.join(cfg.log_dir, "radar_comparison.png")
    bar_path = os.path.join(cfg.log_dir, "f1_comparison.png")
    
    try:
        plot_radar_chart(all_results, save_path=radar_path)
        plot_bar_chart(all_results, save_path=bar_path)
        logger.info(f"saved comparison charts to:\n - {radar_path}\n - {bar_path}")
    except Exception as e:
        logger.warning(f"Failed to create comparison charts. Error: {e}")
    
    # 6. Print summary scoreboard to Terminal
    logger.info("\n" + "="*50 + "\nFINAL RESULTS\n" + "="*50)
    for name, metrics in all_results.items():
        logger.info(f"   MODEL: [{name}]")
        logger.info(f"   Semantic Score : {metrics.get('semantic', 0.0):.4f} | METEOR : {metrics.get('meteor', 0.0):.4f}")
        logger.info(f"   F1-Score       : {metrics.get('f1', 0.0):.4f} | BLEU-4 : {metrics.get('bleu4', 0.0):.4f}")
        logger.info(f"   Accuracy (VQA) : {metrics.get('accuracy', 0.0):.4f} | EM     : {metrics.get('em', 0.0):.4f}\n")

if __name__ == "__main__":
    main()