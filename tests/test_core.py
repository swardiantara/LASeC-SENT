import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.baselines import mask, regex_masked_dedup
from lasec.llm.grouping import validate
from lasec.llm.prompts import build_demonstrations
from lasec.metrics import ga_correct, grouping_report, pairwise_prf, triage_report
from lasec.segment import segment
from lasec.stats import holm, mcnemar_exact
from lasec.train import build_pairs


def test_segment_semicolon_before_pysbd():
    cell = "Data Recorder File Index is 80.; Set Return to Home (RTH) altitude to 100 m (328 ft).; RC signal lost. Returning to home."
    assert segment(cell) == [["Data Recorder File Index is 80."],
                             ["Set Return to Home (RTH) altitude to 100 m (328 ft)."],
                             ["RC signal lost.", "Returning to home."]]


def test_segment_keeps_decimals():
    assert segment("Downlink Restored (after 0m 5.2s).") == [["Downlink Restored (after 0m 5.2s)."]]


def test_mask():
    assert mask("Auto RTH in 3 s.") == mask("Auto RTH in 9 s.") == "auto rth in <*> s"
    assert mask("Max altitude set to nnn") == "max altitude set to <*>"
    g = regex_masked_dedup(["Downlink Restored (after 0m 5.2s).", "Downlink Restored (after 0m 16s)."])
    assert len(set(g.values())) == 1


def test_group_accuracy():
    gold = ["A", "A", "B", "C"]
    assert ga_correct(gold, [1, 1, 2, 3]).tolist() == [1, 1, 1, 1]
    assert ga_correct(gold, [1, 1, 1, 3]).tolist() == [0, 0, 0, 1]    # over-merge breaks all three
    assert ga_correct(gold, [1, 2, 3, 4]).tolist() == [0, 0, 1, 1]    # split breaks A only
    r = grouping_report(gold, [1, 1, 1, 3], list("wxyz"), weights=[1, 1, 2, 4])
    assert r["n_overmerged_clusters"] == 1 and math.isclose(r["ga_weighted"], 4 / 8)


def test_pairwise_prf():
    p, r, f = pairwise_prf(["A", "A", "B"], [1, 1, 1])
    assert math.isclose(p, 1 / 3) and r == 1.0


def test_build_pairs_unordered_once():
    pos, neg = build_pairs(["a", "b", "c", "d"], ["X", "X", "Y", "Z"])
    assert len(pos) == 1 and len(neg) == 5   # C(4,2) = 6 pairs in total


def test_build_pairs_sampled_linear():
    s = [f"s{i}" for i in range(100)]
    pos, neg = build_pairs(s, [f"E{i // 2}" for i in range(100)], negatives="sampled", neg_per_pos=3)
    assert len(pos) == 50 and len(neg) == 300


def test_triage_report_hidden_key_event():
    occ = ["RC signal lost.", "RC signal lost.", "RC signal weak.", "Compass error."]
    gold = {"RC signal lost.": "RC_LOST", "RC signal weak.": "RC_WEAK", "Compass error.": "COMPASS"}
    pred = {"RC signal lost.": 0, "RC signal weak.": 0, "Compass error.": 1}   # over-merge of lost/weak
    r = triage_report(occ, gold, pred, ["RC signal weak.", "Compass error."])
    assert r["n_triage_events"] == 2 and r["key_recovered"] == 1 and r["hidden_key_events"] == ["RC_WEAK"]
    assert math.isclose(r["workload_reduction"], 0.5)


def test_validate_llm_output():
    labels, problems = validate({"groups": [{"event": "E1", "members": [0, 2]}, {"event": "E2", "members": [2, 5]}]}, 4)
    assert labels[0] == labels[2] == "E1" and labels[1].startswith("__UNASSIGNED") and problems


def test_demonstrations_exclude_test_sentences():
    s = [f"s{i}" for i in range(40)]
    e = [f"E{i // 3}" for i in range(40)]
    demo_s, answer = build_demonstrations(s, e, exclude=["s0", "s1"], n_groups=6, seed=1)
    assert "s0" not in demo_s and "s1" not in demo_s
    ids = sorted(i for g in answer["groups"] for i in g["members"])
    assert ids == list(range(len(demo_s)))


def test_holm_and_mcnemar():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert math.isclose(adj["a"], 0.03) and math.isclose(adj["c"], 0.06) and math.isclose(adj["b"], 0.06)
    assert mcnemar_exact(np.ones(5), np.ones(5))["p"] == 1.0


def test_openrouter_request_and_parsing(monkeypatch):
    from types import SimpleNamespace as NS

    import lasec.llm.openrouter_provider as orp
    from lasec.llm.grouping import make_grouper
    sent = {}

    class FakeCompletions:
        def create(self, **kw):
            sent.update(kw)
            msg = NS(content='```json\n{"groups": [{"event": "E", "members": [0, 1]}]}\n```')
            return NS(choices=[NS(message=msg, finish_reason="stop")], model="anthropic/claude-opus-5.5",
                      usage=NS(model_dump=lambda: {"cost": 0.01}), id="gen-1")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(orp, "OpenAI", lambda **kw: NS(chat=NS(completions=FakeCompletions())))
    g = make_grouper({"name": "x", "model": "anthropic/claude-opus-5.5", "enabled": True},
                     {"reasoning_effort": "high", "temperature": None, "max_tokens": 1000})
    r = g("sys", "user")
    assert r["parsed"]["groups"][0]["members"] == [0, 1] and r["usage"]["cost"] == 0.01
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["extra_body"]["reasoning"] == {"effort": "high"} and "temperature" not in sent
    assert sent["extra_body"]["provider"]["require_parameters"] is True


def test_load_dotenv(tmp_path, monkeypatch):
    from lasec.utils import load_dotenv
    f = tmp_path / ".env"
    f.write_text('# comment\nOPENROUTER_API_KEY="sk-or-test"\nEMPTY=\nexport OTHER=1\n', encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)
    load_dotenv(f)
    import os
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-test" and os.environ["OTHER"] == "1" and "EMPTY" not in os.environ
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")
    load_dotenv(f)
    assert os.environ["OPENROUTER_API_KEY"] == "from-shell"


def test_clustering_algorithms_and_order_invariance():
    from lasec.cluster import ALGORITHMS, cluster_labels, cluster_sentences
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 16))
    X = np.vstack([c + 0.05 * rng.normal(size=(5, 16)) for c in centers] + [rng.normal(size=(1, 16)) * 3])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    for alg in ALGORITHMS:
        lab = cluster_labels(X, alg, 0.2)
        assert len(lab) == len(X) and (lab >= 0).all()          # noise points became singletons
    s = [f"s{i:02d}" for i in range(len(X))]
    parts = set()
    for seed in range(5):
        p = cluster_sentences(s, X, 0.2, algorithm="agglomerative", order="shuffle", seed=seed)
        groups = {}
        for k, v in p.items():
            groups.setdefault(v, set()).add(k)
        parts.add(frozenset(frozenset(g) for g in groups.values()))
    assert len(parts) == 1                                      # same partition for every input order
