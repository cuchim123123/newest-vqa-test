"""Full VQA model: CNN + Question LSTM + Answer LSTM with spatial attention."""

from __future__ import annotations
import random
from typing import Optional
import torch
import torch.nn as nn
from src.data.dataset import PAD_IDX, SOS_IDX, EOS_IDX
from src.models.encoder import CNNEncoder, QuestionEncoder
from src.models.decoder import AnswerDecoder

class VQAModel(nn.Module):
    def __init__(
        self, q_vocab_size: int, a_vocab_size: int, embed_size: int = 300,
        hidden_size: int = 512, num_layers: int = 2, dropout: float = 0.3,
        use_pretrained_cnn: bool = True, use_attention: bool = False,
        q_pretrained_emb: Optional[torch.Tensor] = None,
        a_pretrained_emb: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.image_encoder = CNNEncoder(pretrained=use_pretrained_cnn)
        self.question_encoder = QuestionEncoder(q_vocab_size, embed_size, hidden_size, num_layers, dropout, q_pretrained_emb)
        self.answer_decoder = AnswerDecoder(a_vocab_size, embed_size, hidden_size, num_layers, dropout, use_attention, a_pretrained_emb)

    def forward(self, images, questions, q_lengths, answers=None, tf_ratio=0.5, raw_questions=None):
        img_feat = self.image_encoder(images)
        q_out, (h, c), q_mask = self.question_encoder(questions, q_lengths)
        max_t = answers.size(1) - 1
        outputs = []
        for t in range(max_t):
            tok = answers[:, t] if (t == 0 or random.random() < tf_ratio) else pred.argmax(1)
            pred, h, c = self.answer_decoder(tok, h, c, img_feat, q_out, q_mask)
            outputs.append(pred.unsqueeze(1))
        return torch.cat(outputs, 1)

    @torch.no_grad()
    def generate(self, images, questions, q_lengths, use_beam=False, beam_width=5, max_len=40, len_alpha=0.6, rep_penalty=1.2, min_gen_len=5, raw_questions=None):
        img_feat = self.image_encoder(images)
        q_out, (h, c), q_mask = self.question_encoder(questions, q_lengths)
        if use_beam:
            return self._beam_search(img_feat, q_out, q_mask, h, c, beam_width, max_len, len_alpha, rep_penalty, min_gen_len)
        return self._greedy(img_feat, q_out, q_mask, h, c, max_len)

    def _greedy(self, img_feat, q_out, q_mask, h, c, max_len=40):
        B = img_feat.size(0)
        tok = torch.full((B,), SOS_IDX, dtype=torch.long, device=img_feat.device)
        gen = []
        for _ in range(max_len):
            pred, h, c = self.answer_decoder(tok, h, c, img_feat, q_out, q_mask)
            tok = pred.argmax(1)
            gen.append(tok.unsqueeze(1))
            if (tok == EOS_IDX).all(): break
        return torch.cat(gen, 1)

    def _beam_search(self, img_feat, q_out, q_mask, h, c, beam_width=5, max_len=40, len_alpha=0.6, rep_penalty=1.2, min_gen_len=5):
        B = img_feat.size(0)
        device = img_feat.device
        results = []
        _lp = lambda length: ((5 + length) ** len_alpha) / ((5 + 1) ** len_alpha)
        
        for b in range(B):
            im = img_feat[b:b+1]
            qo = q_out[b:b+1] if q_out is not None else None
            qm = q_mask[b:b+1] if q_mask is not None else None
            hb, cb = h[:, b:b+1, :].contiguous(), c[:, b:b+1, :].contiguous()
            
            # (Score, Sequence, h, c)
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
                    lp = torch.log_softmax(pred, -1).squeeze(0)
                    
                    # 🔴 HỆ THỐNG PHẠT LẶP TỪ NÂNG CAO (Advanced Repetition Penalty)
                    if rep_penalty > 1.0:
                        from collections import Counter
                        counts = Counter(seq[1:])
                        for t_id, cnt in counts.items():
                            # Phạt càng nặng nếu từ đó đã xuất hiện càng nhiều lần (Exponential penalty)
                            penalty_factor = rep_penalty ** cnt
                            if lp[t_id] < 0:
                                lp[t_id] *= penalty_factor
                            else:
                                lp[t_id] /= penalty_factor
                                
                        # Phạt siêu nặng (x10) nếu rớt vào vòng lặp lảm nhảm 2 từ liên tiếp giống nhau
                        if len(seq) > 1 and seq[-1] != SOS_IDX:
                            if lp[seq[-1]] < 0:
                                lp[seq[-1]] *= (rep_penalty * 10)
                                
                    # Ngăn chặn việc ngắt câu quá sớm
                    if len(seq) < min_gen_len + 1:
                        lp[EOS_IDX] = -1e4
                        
                    topv, topi = lp.topk(beam_width)
                    for v, i in zip(topv, topi): 
                        cands.append((sc + v.item(), seq + [i.item()], nh, nc))
                        
                if not cands: break
                cands.sort(key=lambda x: x[0] / _lp(len(x[1])), reverse=True)
                beams = cands[:beam_width]
                
                # Chặn sớm nếu tất cả các tia (beams) đều đã chạy tới EOS
                if all(s[1][-1] == EOS_IDX for s in beams): break
                
            # Đẩy nốt các beam chưa chạm EOS vào list done
            for sc, seq, _, _ in beams: 
                if seq[-1] != EOS_IDX: seq.append(EOS_IDX)
                done.append((sc / _lp(len(seq)), seq))
                
            done.sort(key=lambda x: x[0], reverse=True)
            results.append(done[0][1][1:]) # Bỏ SOS token
            
        ml = max(len(s) for s in results)
        padded_results = [s + [PAD_IDX]*(ml-len(s)) for s in results]
        return torch.tensor(padded_results, device=device)