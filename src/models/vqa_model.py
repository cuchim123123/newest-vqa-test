"""Full VQA model: CNN + BiLSTM + GTU Fusion + Dual Attention + Multi-task classification."""

from __future__ import annotations
import random
from collections import Counter
from typing import Optional
import torch
import torch.nn as nn
from src.data.dataset import PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX
from src.models.encoder import CNNEncoder, QuestionEncoder
from src.models.decoder import AnswerDecoder
from src.models.fusion import GatedTanhFusion

class VQAModel(nn.Module):
    """Unified VQA model that combines the best of both project codebases.

    Upgrades over previous version:
    - **BiLSTM** question encoder with LayerNorm and projection.
    - **GTU fusion** in the decoder (replaces raw concatenation).
    - **Multi-task classification head** — auxiliary classifier on top of
      the fused question + image representation for answer prediction.
    """

    def __init__(
        self, q_vocab_size: int, a_vocab_size: int, embed_size: int = 300,
        hidden_size: int = 512, num_layers: int = 2, dropout: float = 0.3,
        use_pretrained_cnn: bool = True, use_attention: bool = False,
        bidirectional: bool = True, num_answers: int = 0,
        q_pretrained_emb: Optional[torch.Tensor] = None,
        a_pretrained_emb: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.hidden_size = hidden_size
        self.num_answers = num_answers

        self.image_encoder = CNNEncoder(pretrained=use_pretrained_cnn)
        self.question_encoder = QuestionEncoder(
            q_vocab_size, embed_size, hidden_size, num_layers, dropout,
            bidirectional=bidirectional, pretrained_emb=q_pretrained_emb,
        )
        self.answer_decoder = AnswerDecoder(
            a_vocab_size, embed_size, hidden_size, num_layers, dropout,
            use_attention, a_pretrained_emb,
        )

        # ── Multi-task classification head ────────────────────────
        # Fuses global image + question representations → answer class
        if num_answers > 0:
            cls_in = hidden_size + CNNEncoder.CNN_OUT_DIM  # question hidden + img pooled
            self.classifier_fusion = GatedTanhFusion(cls_in, hidden_size, dropout)
            self.classifier = nn.Linear(hidden_size, num_answers)
        else:
            self.classifier_fusion = None
            self.classifier = None

    def forward(
        self, images, questions, q_lengths, answers=None,
        tf_ratio=0.5, raw_questions=None,
    ):
        """Forward pass with teacher forcing.

        Returns:
            decoder_outputs: ``(B, T-1, a_vocab_size)`` logits.
            cls_logits:      ``(B, num_answers)`` if classifier is present, else ``None``.
        """
        img_feat = self.image_encoder(images)           # (B, 49, 512)
        q_out, (h, c), q_mask = self.question_encoder(questions, q_lengths)

        # ── Decoder (generative) ─────────────────────────────────────
        max_t = answers.size(1) - 1
        outputs = []
        tok = answers[:, 0]  # Always start with SOS token
        for t in range(max_t):
            pred, h, c = self.answer_decoder(tok, h, c, img_feat, q_out, q_mask)
            outputs.append(pred.unsqueeze(1))
            # Next input: teacher force with ground truth or use own prediction
            if t + 1 < max_t:
                use_tf = random.random() < tf_ratio
                tok = answers[:, t + 1] if use_tf else pred.argmax(1)
        decoder_outputs = torch.cat(outputs, 1)

        # ── Classification head (multi-task) ─────────────────────
        cls_logits = None
        if self.classifier is not None:
            # Global pooled image + last question hidden state
            img_global = img_feat.mean(dim=1)                     # (B, 512)
            q_hidden = h[-1]                                       # (B, H)
            cls_input = torch.cat([q_hidden, img_global], dim=-1)  # (B, H+512)
            cls_fused = self.classifier_fusion(cls_input)          # (B, H)
            cls_logits = self.classifier(cls_fused)                # (B, num_answers)

        return decoder_outputs, cls_logits

    @torch.no_grad()
    def generate(
        self, images, questions, q_lengths,
        use_beam=False, beam_width=5, max_len=30,
        len_alpha=0.7, rep_penalty=1.2, min_gen_len=1,
        raw_questions=None,
        temperature=1.0, suppress_unk=True,
    ):
        img_feat = self.image_encoder(images)
        q_out, (h, c), q_mask = self.question_encoder(questions, q_lengths)
        if use_beam:
            return self._beam_search(
                img_feat, q_out, q_mask, h, c,
                beam_width, max_len, len_alpha, rep_penalty, min_gen_len,
                temperature=temperature, suppress_unk=suppress_unk,
            )
        return self._greedy(img_feat, q_out, q_mask, h, c, max_len)

    def _greedy(self, img_feat, q_out, q_mask, h, c, max_len=30):
        B = img_feat.size(0)
        tok = torch.full((B,), SOS_IDX, dtype=torch.long, device=img_feat.device)
        gen = []
        for _ in range(max_len):
            pred, h, c = self.answer_decoder(tok, h, c, img_feat, q_out, q_mask)
            tok = pred.argmax(1)
            gen.append(tok.unsqueeze(1))
            if (tok == EOS_IDX).all():
                break
        return torch.cat(gen, 1)

    def _beam_search(
        self, img_feat, q_out, q_mask, h, c,
        beam_width=5, max_len=30, len_alpha=0.6,
        rep_penalty=1.2, min_gen_len=5,
        temperature=1.0, suppress_unk=True,
    ):
        B = img_feat.size(0)
        device = img_feat.device
        results = []
        _lp = lambda length: ((5 + length) ** len_alpha) / ((5 + 1) ** len_alpha)

        # Tokens that should never be generated
        _suppress = {PAD_IDX, SOS_IDX}  # PAD and SOS are never valid outputs
        if suppress_unk:
            _suppress.add(UNK_IDX)

        for b in range(B):
            im = img_feat[b:b+1]
            qo = q_out[b:b+1] if q_out is not None else None
            qm = q_mask[b:b+1] if q_mask is not None else None
            hb = h[:, b:b+1, :].contiguous()
            cb = c[:, b:b+1, :].contiguous()

            beams = [(0.0, [SOS_IDX], hb, cb)]
            done = []

            for step in range(max_len):
                cands = []
                for sc, seq, hh, cc in beams:
                    if seq[-1] == EOS_IDX:
                        done.append((sc / _lp(len(seq)), seq))
                        continue

                    tok = torch.tensor([seq[-1]], device=device)
                    pred, nh, nc = self.answer_decoder(tok, hh, cc, im, qo, qm)

                    # Temperature scaling
                    if temperature != 1.0:
                        pred = pred / temperature

                    lp = torch.log_softmax(pred, -1).squeeze(0)

                    # Suppress invalid tokens (PAD, SOS, optionally UNK)
                    for t_id in _suppress:
                        lp[t_id] = -1e9

                    # Repetition penalty
                    if rep_penalty > 1.0:
                        counts = Counter(seq[1:])
                        for t_id, cnt in counts.items():
                            penalty_factor = rep_penalty ** cnt
                            if lp[t_id] < 0:
                                lp[t_id] *= penalty_factor
                            else:
                                lp[t_id] /= penalty_factor

                    # Prevent premature EOS
                    if len(seq) < min_gen_len + 1:
                        lp[EOS_IDX] = -1e4

                    topv, topi = lp.topk(beam_width)
                    for v, i in zip(topv, topi):
                        cands.append((sc + v.item(), seq + [i.item()], nh, nc))

                if not cands:
                    break
                cands.sort(key=lambda x: x[0] / _lp(len(x[1])), reverse=True)
                beams = cands[:beam_width]

                if all(s[1][-1] == EOS_IDX for s in beams):
                    break

            for sc, seq, _, _ in beams:
                if seq[-1] != EOS_IDX:
                    seq.append(EOS_IDX)
                done.append((sc / _lp(len(seq)), seq))

            done.sort(key=lambda x: x[0], reverse=True)
            results.append(done[0][1][1:])  # strip SOS

        ml = max(len(s) for s in results)
        padded = [s + [PAD_IDX] * (ml - len(s)) for s in results]
        return torch.tensor(padded, device=device)