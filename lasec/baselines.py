"""Alternative ways an investigator could obtain event groups from segmented sentences.

Every method maps a list of unique sentences to {sentence: group_id}.
"""
import re
from typing import Dict, Sequence

# numbers (incl. decimals) and the 'nnn'/'xx' placeholders found in the vendor message lists
_NUM = re.compile(r"\d+(?:[.,]\d+)?|\bn{2,}\b|\bx{2,}\b", re.IGNORECASE)


def exact_dedup(sentences: Sequence[str]) -> Dict[str, str]:
    """Current practice (e.g. DroPTC): each distinct string is one event."""
    return {s: s for s in sentences}


def mask(sentence: str) -> str:
    s = _NUM.sub("<*>", sentence)
    return re.sub(r"\s+", " ", s).strip().rstrip(".").lower()


def regex_masked_dedup(sentences: Sequence[str]) -> Dict[str, str]:
    """Mask numeric parameters, then de-duplicate."""
    return {s: mask(s) for s in sentences}


def drain3_groups(sentences: Sequence[str], sim_th: float = 0.4, depth: int = 4,
                  max_children: int = 100) -> Dict[str, int]:
    """Drain (He et al., 2017) via the drain3 package, with numeric masking.
    Sentences are fed in sorted order for determinism."""
    from drain3 import TemplateMiner
    from drain3.masking import MaskingInstruction
    from drain3.template_miner_config import TemplateMinerConfig

    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = sim_th
    cfg.drain_depth = depth
    cfg.drain_max_children = max_children
    cfg.masking_instructions = [MaskingInstruction(r"\d+(?:[.,]\d+)?", "NUM")]
    cfg.profiling_enabled = False
    miner = TemplateMiner(persistence_handler=None, config=cfg)
    for s in sorted(sentences):
        miner.add_log_message(s)
    # second pass: assign every sentence to its final cluster (templates may have generalised)
    return {s: miner.match(s, full_search_strategy="always").cluster_id for s in sentences}
