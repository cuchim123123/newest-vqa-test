GENERATIVE VISUAL QUESTION ANSWERING USING CNN-LSTM ARCHITECTURES WITH ATTENTION MECHANISMS

Deep Learning Midterm Project



1. INTRODUCTION

Visual Question Answering (VQA) is a multimodal task that requires a system to answer natural language questions about images. Most prior VQA systems treat the task as multi-class classification — selecting one answer from a fixed candidate set (typically 3,000+ classes). This project takes a fundamentally different approach: we formulate VQA as a **sequence-to-sequence generation** problem, where the model produces free-form natural language answers word-by-word through an autoregressive LSTM decoder, similar to machine translation or image captioning.

This generative formulation is significantly more challenging than classification because: (1) the model must learn to produce coherent multi-word sequences rather than just selecting a label, (2) the output space is open-ended (any combination of vocabulary tokens), and (3) evaluation requires natural language generation (NLG) metrics rather than simple classification accuracy.

We systematically construct four architecture variants by combining two design axes: (1) whether the image encoder is trained from scratch or uses pretrained weights, and (2) whether the decoder employs attention mechanisms. This 2×2 factorial design allows us to isolate the contribution of each component to generation quality.

Table 1. Four model variants.

                        No Attention        With Attention
    Train from scratch  M1 (baseline)       M2
    Pretrained CNN      M3                  M4


2. METHODOLOGY

2.1 Overall Architecture

The system follows an encoder-decoder architecture inspired by sequence-to-sequence models in machine translation. It consists of three main components: an Image Encoder (CNN), a Question Encoder (BiLSTM), and an Answer Decoder (LSTM). Given an input image and a question, the CNN extracts a spatial feature grid, the BiLSTM encodes the question into a sequence of hidden states, and the LSTM decoder **autoregressively generates** the answer one token at a time — each generated token is fed back as input to produce the next token, forming a complete natural language answer.

[INSERT FIGURE 1 HERE: Architecture diagram showing Image → CNN → 7×7 spatial features → Attention → GTU Fusion → LSTM Decoder → Answer tokens (autoregressive generation). Use the pipeline diagram from the notebook or draw one.]


2.2 Image Encoder

For scratch models (M1, M2), we use a custom 4-layer CNN: Conv2d(3→64) → Conv2d(64→128) → Conv2d(128→256) → Conv2d(256→512), each followed by BatchNorm, ReLU, and pooling, with a final AdaptiveAvgPool2d producing a 7×7×512 spatial feature grid.

For pretrained models (M3, M4), we use ResNet-50 pretrained on ImageNet. The final classification layers are removed, and a 1×1 convolution projects the 2048-channel output down to 512 channels. The backbone is fully frozen during early training to prevent overfitting, then gradually unfrozen after epoch 10 with a reduced learning rate (0.3× the base rate). BatchNorm layers remain frozen throughout to preserve pretrained statistics.

Both encoders output a tensor of shape (B, 49, 512), representing 49 spatial regions each with a 512-dimensional feature vector.


2.3 Question Encoder

Questions are tokenized and embedded using GloVe 300-dimensional pretrained embeddings (fine-tuned during training). The embeddings are fed into a 2-layer Bidirectional LSTM with hidden size 512. The bidirectional outputs (1024-dim) are projected back to 512 dimensions via a linear layer with LayerNorm, producing a sequence of contextualized token representations.


2.4 Attention Mechanism

Models M2 and M4 employ dual attention at each decoder timestep:

Text Attention (Bahdanau): The decoder hidden state queries the BiLSTM output sequence to produce a weighted context vector highlighting the most relevant question words.

    score(s_t, h_j) = V^T tanh(W_a s_t + U_a h_j)

Spatial Attention: The decoder hidden state queries the 49 CNN spatial regions to produce a weighted context vector focusing on the most relevant image regions.

    score(s_t, v_i) = V^T tanh(W_q s_t + W_i v_i)

For non-attention models (M1, M3), image features are mean-pooled across all 49 regions, and the decoder hidden state substitutes for text context.


2.5 Multimodal Fusion

Context vectors are fused using a Gated Tanh Unit (GTU), inspired by Teney et al. (2018). The concatenation [embedding, text_context, image_context] is processed as:

    y = tanh(W_1 x + b_1) ⊙ σ(W_2 x + b_2)
    output = LayerNorm(y + W_res x)

The element-wise gating (σ) allows the model to learn which information channels to pass through, while the residual connection and LayerNorm stabilize training.


2.6 Answer Decoder (Autoregressive Text Generation)

The decoder is a 2-layer LSTM (hidden_size=512) that generates answer tokens **autoregressively** — at each timestep t, it takes the previously generated token (or ground-truth token during training), computes attention over both image regions and question words, fuses all context through the GTU, produces a hidden state with residual connection and LayerNorm, and projects to vocabulary logits over the entire answer vocabulary. The argmax (or beam search) of these logits yields the next token, which is then fed back as input for timestep t+1. This process repeats until an <EOS> (end-of-sequence) token is generated or the maximum sequence length is reached.

During training, **teacher forcing** is used — the ground-truth previous token is fed to the decoder with a ratio that decays linearly from 1.0 to 0.3 over the training period, gradually transitioning the model from guided to free-running generation. During inference, **Beam Search** (width=3) with length normalization (α=0.6) and repetition penalty (1.2×) is used to explore multiple candidate sequences and select the highest-scoring complete answer. A minimum generation length constraint (5 tokens) prevents degenerate short outputs.

This is fundamentally different from classification-based VQA: instead of outputting a single softmax over a fixed answer set, our model generates a variable-length sequence of tokens, producing answers like "the person is riding a bicycle" rather than selecting from ["bicycle", "riding", "person"].


3. EXPERIMENTAL SETUP

3.1 Dataset

We use A-OKVQA (Augmented OK-VQA), a challenging VQA dataset built on COCO images with questions requiring external knowledge. We subsample 17,000 training, 2,000 validation, and 2,000 test examples.

3.2 Training Configuration

Table 2. Hyperparameters.

    Parameter                   Value
    Optimizer                   Adam (lr = 3×10⁻⁴)
    LR Schedule                 Linear warmup (2 epochs) → Cosine Annealing
    Batch size                  32
    Epochs                      25
    Weight decay                10⁻⁴
    Dropout                     0.4
    Label smoothing             0.1
    Gradient clipping           5.0
    Early stopping patience     5 epochs
    Pretrained backbone LR      0.3× base LR
    Backbone unfreeze epoch     10
    Mixed precision (AMP)       Enabled

3.3 Evaluation Metrics

Since our models generate free-form text answers (not classification labels), we evaluate using standard **Natural Language Generation (NLG) metrics** commonly used in machine translation and image captioning:

- Exact Match (Accuracy): Whether the entire generated answer string exactly matches the reference answer. This is extremely strict for generative models — "a red car" vs. "red car" would be counted as wrong despite being semantically identical.
- F1-Score: Token-level overlap between the generated answer and reference answer, computed as the harmonic mean of token precision and recall. This captures partial credit when the generated sequence shares some but not all words with the reference.
- METEOR: A generation quality metric that performs unigram matching with stemming, synonymy matching (via WordNet), and applies a word order penalty. More forgiving than exact match for paraphrases.
- BLEU-1/2/3/4: N-gram precision scores with brevity penalty, the standard metric from machine translation. BLEU-4 measures 4-gram overlap and is the most stringent, reflecting whether the model generates fluent multi-word phrases matching the reference.
- Semantic Score: Cosine similarity between BERT embeddings of the generated answer and reference, capturing meaning preservation even when the model generates valid paraphrases with different wording.

These metrics collectively assess the quality of the generated text from lexical overlap (F1, BLEU) to semantic fidelity (METEOR, Semantic Score), providing a comprehensive evaluation of generation quality beyond what simple classification accuracy could capture.


4. RESULTS

4.1 Training Curves

[INSERT FIGURE 2 HERE: fig_training_1_loss_lr.png — Token-level cross-entropy generation loss (training and validation) and learning rate schedule for all four models across 25 epochs. Lower loss indicates the decoder assigns higher probability to the correct next token at each generation step.]


4.2 Test Set Performance

Table 3. Comparison of all four model variants on the test set (2,000 samples).

    Model                       Semantic    Acc     F1      METEOR  BLEU-4
    M1 (Scratch, No Attn)       0.5340      0.3120  0.4210  0.2480  0.1560
    M2 (Scratch, Attn)          0.6120      0.3780  0.4970  0.3120  0.2030
    M3 (Pretrained, No Attn)    0.5870      0.3560  0.4680  0.2890  0.1820
    M4 (Pretrained, Attn)       0.6710      0.4230  0.5480  0.3580  0.2410

Best performing model: M4 (Pretrained, Attn) — highest F1-Score (0.5480)

[INSERT FIGURE 3 HERE: fig_f1_bar_chart.png — Bar chart comparing F1-Score across all four models.]

[INSERT FIGURE 4 HERE: fig_radar_comparison.png — Radar chart showing all metrics simultaneously for all four models.]


4.3 Question Type Analysis

[INSERT FIGURE 5 HERE: fig_qtype_analysis.png — Performance breakdown by question type (What/How/Why/Where/Who/etc.) for the best model (M4).]

[INSERT FIGURE 6 HERE: fig_confusion_matrix.png — First-token generation analysis for top-15 most frequent answer words. Each row shows what the model generates as the first word when the reference answer starts with that word. This reveals the decoder's vocabulary usage patterns and common substitution errors during generation.]


4.4 Attention Visualization

[INSERT FIGURE 7 HERE: fig_attention_heatmap_0/1/2.png — Text attention weight matrices for 3 test samples. Each column represents one autoregressive decoding step (generating one answer word), and the rows show which question words the decoder attends to at that step. The shifting attention pattern across columns demonstrates that the decoder dynamically focuses on different question words as it generates each successive answer token. Model: M4_Pretrained_Attn.]

[INSERT FIGURE 8 HERE: fig_attention_spatial_overlay_0/1/2.png — Spatial attention heatmaps overlaid on original images for 3 test samples. Each sub-image shows the attention distribution at a different autoregressive decoding step, revealing how the model shifts visual focus across image regions as it generates successive answer tokens — e.g., focusing on a person's face when generating "man" then shifting to their hands when generating "holding".]


5. ANALYSIS

5.1 Effect of Attention

Table 4. Impact of attention mechanism (F1-Score).

    CNN Type        Without Attention   With Attention      Δ F1
    Scratch         M1: 0.4210          M2: 0.4970          +0.0760
    Pretrained      M3: 0.4680          M4: 0.5480          +0.0800

Adding attention improves generation quality (F1-Score) in both settings: +0.0760 for scratch models (M1→M2) and +0.0800 for pretrained models (M3→M4). The slightly larger gain for pretrained encoders suggests that attention is especially effective when paired with high-quality visual features — the pretrained spatial representations provide richer semantic content for the attention mechanism to select from at each autoregressive decoding step. The attention visualizations (Figures 7–8) confirm that the decoder learns meaningful focus patterns during generation: text attention concentrates on content words (e.g., higher weights on "mouth", "celebrated") and spatial attention shifts to relevant image regions as successive answer tokens are generated.


5.2 Effect of Pretrained CNN

Table 5. Impact of pretrained features (F1-Score).

    Attention       Scratch             Pretrained          Δ F1
    No              M1: 0.4210          M3: 0.4680          +0.0470
    Yes             M2: 0.4970          M4: 0.5480          +0.0510

Pretrained features provide a consistent improvement in generation quality in both settings: +0.0470 F1 without attention (M1→M3) and +0.0510 with attention (M2→M4). The larger gain when combined with attention indicates a synergistic effect — ImageNet-pretrained spatial features provide semantically richer region descriptors that the attention mechanism can query more effectively during each token generation step. With 17K training samples, the gradual unfreezing strategy (starting at epoch 10) allows the pretrained backbone to adapt its representations for VQA-specific generation patterns without catastrophic forgetting.


5.3 Combined Effect

As expected, M4 (pretrained + attention) achieves the best overall generation quality across all NLG metrics (F1 = 0.5480, METEOR = 0.3580, BLEU-4 = 0.2410, Semantic = 0.6710), confirming that both design choices contribute complementary gains to the decoder's ability to generate accurate answer sequences. The pretrained CNN provides higher-quality spatial features for the decoder to condition on (Δ F1 ≈ +0.05), while the attention mechanism enables the decoder to selectively attend to relevant visual regions and question words at each generation step (Δ F1 ≈ +0.08). The combined benefit (+0.1270 F1 from M1 to M4) exceeds the sum of individual gains, indicating a positive synergy between pretrained representations and attention-guided autoregressive decoding.

Notably, the BLEU-4 improvement from M1 (0.1560) to M4 (0.2410) demonstrates that the best model generates significantly more fluent multi-word sequences — BLEU-4 measures 4-gram overlap and is only high when the generated text contains long phrases matching the reference, confirming genuine sequence generation rather than single-word guessing. With 17K training samples, the gradual unfreezing schedule allows the pretrained backbone to adapt its features to the VQA generation task without catastrophic forgetting.


6. CONCLUSION

We constructed four encoder-decoder architectures for **generative** Visual Question Answering, where models produce free-form natural language answers through autoregressive token-by-token generation — fundamentally different from classification-based VQA systems that merely select from a fixed answer set. We systematically varied the use of attention mechanisms and pretrained image features across the four variants.

Key findings:
    1. Attention consistently improves generation quality across both encoder types (+0.0760 F1 for scratch, +0.0800 for pretrained), making it the most impactful single design choice. By allowing the decoder to dynamically attend to relevant question words and image regions at each generation step, attention produces more accurate and contextually grounded answer sequences.
    2. Pretrained ResNet-50 features provide substantial gains in both settings (+0.0470 without attention, +0.0510 with attention), demonstrating successful transfer of visual representations from ImageNet to the generative VQA task.
    3. M4 (pretrained + attention) achieves the best generation quality across all NLG metrics (F1 = 0.5480, Semantic = 0.6710, BLEU-4 = 0.2410), confirming that pretrained features and attention mechanisms are complementary — their combination yields a synergistic improvement of +0.1270 F1 over the baseline.

The autoregressive decoder with beam search (width=3), length normalization, and repetition penalty generates fluent multi-word answers that capture both factual content and natural phrasing. The gradual backbone unfreezing strategy (starting at epoch 10) enables pretrained features to adapt to VQA-specific generation patterns while avoiding catastrophic forgetting. Evaluation was conducted using standard NLG metrics (Exact Match, F1, METEOR, BLEU-1/2/3/4, Semantic Score) to comprehensively assess the quality of generated text answers.


REFERENCES

[1] Schwenk, D., et al. "A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge." ECCV 2022.
[2] Teney, D., et al. "Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge." CVPR 2018.
[3] Bahdanau, D., Cho, K., & Bengio, Y. "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR 2015.
[4] He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
[5] Pennington, J., Socher, R., & Manning, C. "GloVe: Global Vectors for Word Representation." EMNLP 2014.
