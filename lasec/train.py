"""Supervised contrastive fine-tuning of a sentence encoder on sentence-level event labels.

Pairs are built over UNIQUE sentences; every unordered pair is counted once:
  positive (label 1): both sentences have the same event label
  negative (label 0): different event labels
With `negatives: all` this is the complete pair set; `negatives: sampled` keeps at most
`neg_per_pos` negatives per positive (and per sentence when an event has one member),
which keeps the pair count linear in the number of sentences.
"""
import itertools
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def build_pairs(sentences, events, negatives="all", neg_per_pos=5, seed=42):
    assert len(sentences) == len(events) == len(set(sentences)), "pass unique sentences"
    by_event = defaultdict(list)
    for s, e in zip(sentences, events):
        by_event[e].append(s)
    pos = [(a, b, 1.0) for members in by_event.values() for a, b in itertools.combinations(members, 2)]
    if negatives == "all":
        neg = [(a, b, 0.0) for (ea, ma), (eb, mb) in itertools.combinations(by_event.items(), 2)
               for a in ma for b in mb]
    elif negatives == "sampled":
        rng = random.Random(seed)
        ev = list(by_event)
        budget = max(len(pos) * neg_per_pos, len(sentences) * neg_per_pos)
        seen, neg = set(), []
        while len(neg) < budget and len(seen) < math.comb(len(sentences), 2):
            ea, eb = rng.sample(ev, 2)
            a, b = rng.choice(by_event[ea]), rng.choice(by_event[eb])
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key)
                neg.append((a, b, 0.0))
    else:
        raise ValueError(negatives)
    return pos, neg


def finetune(base_model, sentences, events, out_dir, *, epochs=3, lr=2e-5, batch_size=64, margin=0.5,
             warmup_ratio=0.1, negatives="all", neg_per_pos=5, seed=42, device=None, max_pairs=None,
             log=print):
    """Fine-tune `base_model` with the contrastive loss (Hadsell et al., 2006) on cosine distance."""
    import torch
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.util import batch_to_device
    from transformers import get_linear_schedule_with_warmup

    from .utils import set_seed
    set_seed(seed)
    pos, neg = build_pairs(sentences, events, negatives, neg_per_pos, seed)
    pairs = pos + neg
    if max_pairs:  # smoke tests only
        pairs = random.Random(seed).sample(pairs, min(max_pairs, len(pairs)))
    log(f"pairs: {len(pos)} positive, {len(neg)} negative, training on {len(pairs)}")

    model = SentenceTransformer(base_model, device=device)
    loss_fn = losses.ContrastiveLoss(model, distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
                                     margin=margin)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * math.ceil(len(pairs) / batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(warmup_ratio * steps), steps)
    gen = np.random.default_rng(seed)
    model.train()
    history = []
    for ep in range(epochs):
        order = gen.permutation(len(pairs))
        total = 0.0
        for k in range(0, len(order), batch_size):
            batch = [pairs[i] for i in order[k:k + batch_size]]
            a = batch_to_device(model.tokenize([p[0] for p in batch]), model.device)
            b = batch_to_device(model.tokenize([p[1] for p in batch]), model.device)
            y = torch.tensor([p[2] for p in batch], dtype=torch.float, device=model.device)
            loss = loss_fn([a, b], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(batch)
        history.append(total / len(pairs))
        log(f"epoch {ep + 1}/{epochs} loss {history[-1]:.5f}")
    model.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    return dict(n_pos=len(pos), n_neg=len(neg), n_train_pairs=len(pairs), loss_per_epoch=history)
