"""Run an LLM as an event-grouping baseline, with validation, batching and a response cache.

Every raw response is stored under the cache directory; re-running with the cache present
reproduces the reported numbers without calling any API.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .prompts import SYSTEM, user_prompt


def make_grouper(model_cfg: dict, defaults: dict):
    """One OpenRouter client per model; per-model settings override the shared defaults."""
    from .openrouter_provider import OpenRouterGrouper
    kw = {**defaults, **{k: v for k, v in model_cfg.items() if k not in ("name", "enabled")}}
    return OpenRouterGrouper(**kw)


def validate(parsed: dict, n: int):
    """Return (labels per id, problems). Missing ids become singletons; duplicates keep the first."""
    labels, problems = [None] * n, []
    for g in parsed.get("groups", []):
        for i in g.get("members", []):
            if not isinstance(i, int) or not 0 <= i < n:
                problems.append(f"invalid id {i}")
            elif labels[i] is not None:
                problems.append(f"id {i} assigned twice")
            else:
                labels[i] = str(g["event"]).strip()
    missing = [i for i, v in enumerate(labels) if v is None]
    if missing:
        problems.append(f"missing ids {missing}")
        for i in missing:
            labels[i] = f"__UNASSIGNED_{i}"
    return labels, problems


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def llm_group(sentences, grouper, cache_dir, run_id, demo=None, batch_size=100, seed=0, max_retries=2, log=print):
    """Group `sentences`; presentation order is a seeded permutation (seed = run index), so repeated
    runs also probe order sensitivity. Returns ({sentence: event}, run record)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    order = list(np.random.default_rng(seed).permutation(len(sentences)))
    ordered = [sentences[i] for i in order]
    result, known, record = {}, {}, dict(run_id=run_id, seed=seed, batches=[])
    for b0 in range(0, len(ordered), batch_size):
        batch = ordered[b0:b0 + batch_size]
        user = user_prompt(batch, demo, known if b0 else None)
        f = cache_dir / f"{run_id}_b{b0 // batch_size}_{_key(SYSTEM, user)}.json"
        if f.exists():
            resp = json.loads(f.read_text(encoding="utf-8"))
        else:
            for attempt in range(max_retries + 1):
                try:
                    resp = grouper(SYSTEM, user)
                    break
                except (json.JSONDecodeError, RuntimeError) as e:
                    log(f"  attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries:
                        raise
            f.write_text(json.dumps(dict(system=SYSTEM, user=user, **resp), ensure_ascii=False, indent=1),
                         encoding="utf-8")
        labels, problems = validate(resp["parsed"], len(batch))
        if problems:
            log(f"  batch {b0 // batch_size}: {problems}")
        for s, lab in zip(batch, labels):
            result[s] = lab
            known.setdefault(lab, s)
        record["batches"].append(dict(file=f.name, n=len(batch), problems=problems, served_by=resp.get("served_by")))
    return result, record
