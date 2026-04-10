"""Verify if the trimmed vocab mapping matches what the model expects."""
import torch
import sys
sys.path.insert(0, ".")

# Load the current deploy artifact
artifact = torch.load("vqa_deploy_all_models.pth", map_location="cpu", weights_only=False)
q_vocab = artifact["q_vocab"]
a_vocab = artifact["a_vocab"]

print("=== Question Vocab (first 20 tokens) ===")
for i in range(min(20, len(q_vocab))):
    print(f"  {i:4d} → {q_vocab.itos[i]}")

print(f"\n=== Answer Vocab (first 30 tokens) ===")
for i in range(min(30, len(a_vocab))):
    print(f"  {i:4d} → {a_vocab.itos[i]}")

# Check some common words
print("\n=== Spot checks ===")
for word in ["what", "color", "is", "the", "red", "blue", "green", "yes", "no", "dog", "cat"]:
    qi = q_vocab.stoi.get(word, "MISSING")
    ai = a_vocab.stoi.get(word, "MISSING")
    print(f"  '{word}': q_idx={qi}, a_idx={ai}")

# Check what the model actually predicts — look at embedding weights
# The model's answer_decoder.fc.bias can tell us which output indices are most likely
states = artifact["model_states"]
for name in ["M4_Pretrained_Attn"]:
    sd = states[name]
    fc_bias = sd["answer_decoder.fc.bias"]
    # Top 20 most-biased output indices
    top_indices = fc_bias.topk(20).indices.tolist()
    print(f"\n=== {name}: Top 20 biased answer indices ===")
    for idx in top_indices:
        word = a_vocab.itos.get(idx, f"<OOV_{idx}>")
        print(f"  idx={idx:5d} → '{word}' (bias={fc_bias[idx]:.3f})")
