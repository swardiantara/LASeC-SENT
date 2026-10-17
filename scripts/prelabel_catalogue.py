"""Annotation support: propose sentence-level event labels for the catalogue and write
correction sheets (semi-automatic labelling: pseudo-label -> expert correction).

Proposals (in order of precedence):
  1. FPV link     the normalised sentence equals an FPV sentence -> reuse the expert FPV event ID
  2. exact-norm   identical after masking numbers, case, 'Warning:'/'Please' prefixes, punctuation
  3. old-label    both sentences were complete single-sentence messages with the same old message-level event
  4. embedding    agglomerative clustering (cosine, average linkage) of pre-trained embeddings,
                  never across sentence families (status / mode change / setting / instruction / record)
Outputs:
  annotation/catalogue_correction_sheet.xlsx   expert sheet (proposals pre-filled, edit 'final_event')
  annotation/catalogue_iaa_annotator2.xlsx     blind 20% subset for the second annotator
  annotation/fpv_iaa_annotator2_blind.xlsx     blind FPV sheet for the second annotator
"""
import argparse
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.baselines import mask
from lasec.utils import ROOT, load_config, path

INSTR_VERBS = {"adjust", "avoid", "calibrate", "change", "charge", "check", "clean", "connect", "contact", "do",
               "ensure", "fly", "hold", "increase", "insert", "install", "keep", "land", "lightly", "lower", "make",
               "move", "place", "power", "press", "pull", "push", "raise", "reconnect", "reduce", "remove", "replace",
               "reposition", "restart", "retry", "select", "slowly", "step", "stop", "switch", "tap", "toggle", "try",
               "turn", "update", "use", "verify", "wait", "if", "after", "when", "please", "go", "rotate", "re-pair",
               "return", "exit", "release", "reinsert", "recalibrate", "unlock", "unfold", "relaunch", "attach",
               "detach", "format", "pair", "fasten", "drop", "enable", "disable", "operate", "view", "manually"}
AMBIGUOUS_FIRST = {"return", "exit", "switch", "stop", "enable", "disable", "land"}
SUBJECT_AFTER = {"return": {"to", "the", "home", "sticks", "aircraft"}, "land": {"the", "aircraft", "immediately", "promptly", "or", "as", "now"}}


def norm_key(s: str) -> str:
    s = re.sub(r"^\s*(warning|caution|notice|note)\s*:\s*", "", s, flags=re.I)
    s = re.sub(r"^\s*please\s+", "", s, flags=re.I)
    s = mask(s)
    return re.sub(r"[^\w<>*]+", " ", s).strip()


def family(s: str) -> tuple[str, bool]:
    """(family hint, uncertain?)"""
    t = re.sub(r"^\s*(warning|caution|notice)\s*:\s*", "", s, flags=re.I).strip()
    low = t.lower()
    words = re.findall(r"[a-z][a-z\-']*", low)
    first = words[0] if words else ""
    if low.startswith("flight mode changed to") or re.match(r"^(switched|changed) to .* mode", low):
        return "mode_change", False
    if low.startswith("data recorder file index"):
        return "record", False
    if re.match(r"^(set|setting)\b", low) or re.search(r"\b(adjusted|set|changed) to\b", low):
        return "setting", False
    if first in INSTR_VERBS:
        if first in SUBJECT_AFTER and len(words) > 1 and words[1] not in SUBJECT_AFTER[first]:
            return "status", True
        return "instruction", first in AMBIGUOUS_FIRST
    return "status", False


ANTONYMS = [("close", "far"), ("on", "off"), ("paused", "resumed"), ("pause", "resume"), ("large", "small"),
            ("enabled", "disabled"), ("enable", "disable"), ("locked", "unlocked"), ("lost", "restored"),
            ("lost", "recovered"), ("weak", "strong"), ("high", "low"), ("upward", "downward"),
            ("forward", "backward"), ("left", "right"), ("entering", "exiting"), ("enter", "exit"),
            ("started", "stopped"), ("start", "stop"), ("connected", "disconnected"), ("open", "closed"),
            ("raised", "lowered"), ("ascending", "descending"), ("increase", "decrease"), ("succeeded", "failed"),
            ("too fast", "too slow"), ("above", "below"), ("maximum", "minimum"), ("max", "min")]
NEGATIONS = {"not", "no", "unable", "cannot", "can't", "failed", "fail", "unavailable", "never", "error"}


def polarity_conflict(a: str, b: str) -> bool:
    """True if two sentences differ by an antonym pair or by a negation word (e.g. 'returning home' vs
    'unable to return home'). Embedding similarity is unreliable for such pairs, so they are never merged
    automatically."""
    wa, wb = set(re.findall(r"[a-z']+", a.lower())), set(re.findall(r"[a-z']+", b.lower()))
    if (wa & NEGATIONS) != (wb & NEGATIONS):
        return True
    for x, y in ANTONYMS:
        xs, ys = set(x.split()), set(y.split())
        if (xs <= wa and ys <= wb and not ys <= wa) or (ys <= wa and xs <= wb and not xs <= wa):
            return True
    return False


def slug(text: str, prefix: str = "") -> str:
    stop = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for", "be", "has", "been", "your",
            "please", "warning", "it", "this", "with", "will", "may", "can", "not", "no"}
    words = [w for w in re.findall(r"[A-Za-z]+", text) if w.lower() not in stop][:5] or ["EVENT"]
    return prefix + "_".join(w.upper() for w in words)


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2",
                    help="pre-trained encoder for proposals only (never fine-tuned on these labels)")
    ap.add_argument("--merge-distance", type=float, default=0.15, help="cosine distance for embedding proposals")
    ap.add_argument("--near-distance", type=float, default=0.20, help="flag groups whose nearest other group is closer")
    ap.add_argument("--iaa-fraction", type=float, default=0.20)
    ap.add_argument("--force-fpv-blind", action="store_true", help="overwrite an existing blind FPV sheet")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    proc = path(cfg, "processed")
    occ = pd.read_csv(proc / "catalogue_occurrences.csv", keep_default_na=False)
    sents = sorted(occ.sentence.unique())
    n = len(sents)
    idx = {s: i for i, s in enumerate(sents)}
    counts = occ.sentence.value_counts()

    # context per sentence
    msg_n = occ.groupby("line_id").sentence.transform("size")
    occ["single"] = msg_n == 1
    ctx = occ.groupby("sentence").agg(source_message=("message", "first"),
                                      old_events=("old_event_id", lambda x: ", ".join(sorted(set(x) - {""}))),
                                      first_in_message=("sent_pos", lambda x: bool((x == 0).any())))
    fam = {s: family(s) for s in sents}
    # DroPTC problem type (context-dependent for a few sentences, then joined with '/')
    ptype = occ.groupby("sentence").problem_type.agg(lambda x: "/".join(sorted(set(x)))).to_dict()

    # FPV expert labels
    fpv = pd.read_csv(path(cfg, "fpv_labels"))
    fpv_by_key = {}
    for s, e in zip(fpv.sentence, fpv[cfg["labels"]["fpv_column"]]):
        fpv_by_key.setdefault(norm_key(s), e)

    uf, basis = UF(n), ["embedding"] * n
    by_key = defaultdict(list)
    for s in sents:
        by_key[norm_key(s)].append(idx[s])
    for members in by_key.values():
        for m in members[1:]:
            uf.union(m, members[0])
    single_old = defaultdict(list)
    for _, r in occ[occ.single & (occ.old_event_id != "")].iterrows():
        single_old[(r.old_event_id, fam[r.sentence][0])].append(idx[r.sentence])
    for members in single_old.values():
        for m in members[1:]:
            uf.union(m, members[0])

    # embedding proposals over high-precision groups, within family AND DroPTC problem type
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    enc = SentenceTransformer(a.encoder, device="cpu")
    X = enc.encode(sents, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    roots = sorted({uf.find(i) for i in range(n)})
    gmembers = defaultdict(list)
    for i in range(n):
        gmembers[uf.find(i)].append(i)
    gfam = {r: Counter(f"{fam[sents[i]][0]}|{ptype[sents[i]]}" for i in gmembers[r]).most_common(1)[0][0]
            for r in roots}
    final = {}
    for f in sorted(set(gfam.values())):
        rs = [r for r in roots if gfam[r] == f]
        C = np.stack([X[gmembers[r]].mean(0) for r in rs])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        lab = (AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                       distance_threshold=a.merge_distance).fit_predict(C)
               if len(rs) > 1 else [0])
        by_lab = defaultdict(list)
        for r, l in zip(rs, lab):
            by_lab[l].append(r)
        for l, members_roots in by_lab.items():
            # split embedding clusters so that no two merged groups have a polarity conflict
            subs = []
            for r in members_roots:
                for sub in subs:
                    if not any(polarity_conflict(sents[i], sents[j]) for q in sub for i in gmembers[q] for j in gmembers[r]):
                        sub.append(r)
                        break
                else:
                    subs.append([r])
            for k, sub in enumerate(subs):
                for r in sub:
                    for i in gmembers[r]:
                        final[i] = f"{f}:{l}.{k}"
    groups = defaultdict(list)
    for i, g in final.items():
        groups[g].append(i)

    # name groups, mark basis
    names, used = {}, Counter()
    for g, members in groups.items():
        fpv_ids = Counter(fpv_by_key[norm_key(sents[i])] for i in members if norm_key(sents[i]) in fpv_by_key)
        f = g.split(":")[0].split("|")[0]
        if fpv_ids:
            name = fpv_ids.most_common(1)[0][0]
        else:
            rep = sorted((sents[i] for i in members), key=lambda s: (-counts[s], len(s)))[0]
            name = slug(rep, {"instruction": "INSTR_", "mode_change": "MODE_", "setting": "SET_"}.get(f, ""))
        used[name] += 1
        names[g] = name if used[name] == 1 else f"{name}_{used[name]}"
        for i in members:
            k = norm_key(sents[i])
            if k in fpv_by_key:
                basis[i] = "FPV link"
            elif len(by_key[k]) > 1:
                basis[i] = "exact-norm"
            elif any(len(v) > 1 and i in v for v in single_old.values()):
                basis[i] = "old-label"
            elif len(members) > 1:
                basis[i] = "embedding"
            else:
                basis[i] = "singleton"

    # nearest other group (merge candidates) and review flags
    gl = list(groups)
    G = np.stack([X[groups[g]].mean(0) for g in gl])
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    D = 1 - G @ G.T
    np.fill_diagonal(D, np.inf)
    base_names = defaultdict(set)
    for nm in names.values():
        base_names[re.sub(r"_\d+$", "", nm)].add(nm)
    rows = []
    for gi, g in enumerate(gl):
        j = int(np.argmin(D[gi]))
        members = sorted(groups[g], key=lambda i: sents[i])
        mixed_old = len({e for i in members for e in ctx.loc[sents[i], "old_events"].split(", ")}) > 1
        for i in members:
            s = sents[i]
            flags = []
            if fam[s][1]:
                flags.append("family uncertain")
            if fam[s][0] != g.split(":")[0].split("|")[0]:
                flags.append("family differs from group")
            if D[gi, j] < a.near_distance:
                flags.append("check merge with nearest")
            if len(members) > 1 and basis[i] == "embedding":
                flags.append("embedding merge")
            if len(members) > 1 and mixed_old:
                flags.append("members had different old labels")
            if len({ptype[sents[k]] for k in members}) > 1:
                flags.append("members have different DroPTC problem types")
            twins = sorted(base_names[re.sub(r"_\d+$", "", names[g])] - {names[g]})
            if twins:
                flags.append("same-name group: " + ", ".join(twins))
            if len(s.split()) <= 2 and not ctx.loc[s, "first_in_message"]:
                flags.append("possible fragment")
            rows.append(dict(sentence=s, occurrences=int(counts[s]), family_hint=fam[s][0], problem_type=ptype[s],
                             proposed_event=names[g], basis=basis[i], group_size=len(members),
                             review_flags="; ".join(flags), nearest_other_event=names[gl[j]],
                             nearest_distance=round(float(D[gi, j]), 3),
                             nearest_example=sents[sorted(groups[gl[j]], key=lambda k: sents[k])[0]],
                             source_message=ctx.loc[s, "source_message"], old_message_events=ctx.loc[s, "old_events"]))
    df = pd.DataFrame(rows)
    fam_order = {"status": 0, "instruction": 1, "mode_change": 2, "setting": 3, "record": 4}
    df["_f"] = df.proposed_event.map(lambda e: fam_order.get(df.loc[df.proposed_event == e, "family_hint"].mode()[0], 9))
    df = df.sort_values(["_f", "proposed_event", "sentence"]).drop(columns="_f").reset_index(drop=True)
    df.insert(0, "sent_id", [f"C{i + 1:04d}" for i in range(len(df))])

    # stratified IAA subset (by family and by singleton vs multi-member proposal)
    rng = np.random.default_rng(cfg["seed"])
    strata = df.family_hint + "|" + (df.group_size > 1).astype(str)
    pick = []
    for _, sub in df.groupby(strata):
        k = max(1, int(round(a.iaa_fraction * len(sub))))
        pick += list(rng.choice(sub.index, size=min(k, len(sub)), replace=False))
    df["iaa_subset"] = np.where(df.index.isin(pick), "Y", "")

    from annotation_sheets import write_correction_sheet, write_blind_sheet  # noqa: E402
    out = ROOT / "annotation"
    sheet = out / "catalogue_correction_sheet.xlsx"
    df["final_event"], df["notes"] = df["proposed_event"], ""
    if sheet.exists():
        archived = archive(sheet)
        carried = carry_over_edits(archived, df)
        print(f"archived previous sheet to {archived.relative_to(ROOT)}; carried over {carried} expert decisions")
    write_correction_sheet(df, fpv, cfg["labels"]["fpv_column"], sheet)
    iaa_sheet = out / "catalogue_iaa_annotator2.xlsx"
    if iaa_sheet.exists():
        archive(iaa_sheet)
    write_blind_sheet(df[df.iaa_subset == "Y"][["sent_id", "sentence", "source_message"]].sample(frac=1, random_state=cfg["seed"]),
                      iaa_sheet, "catalogue")
    fpv_sheet = out / "fpv_iaa_annotator2_blind.xlsx"
    if not fpv_sheet.exists() or a.force_fpv_blind:
        f2 = fpv[["sentence"]].sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
        f2.insert(0, "sent_id", [f"F{i + 1:03d}" for i in range(len(f2))])
        f2["source_message"] = ""
        write_blind_sheet(f2, fpv_sheet, "FPV")
    print(df.basis.value_counts().to_dict(), "| proposed events:", df.proposed_event.nunique(),
          "| flagged rows:", int((df.review_flags != "").sum()), "| IAA subset:", len(pick))


def archive(file: Path) -> Path:
    """Move an existing sheet to annotation/archive/<name>_<timestamp>.xlsx (never overwrite work)."""
    dst = file.parent / "archive" / f"{file.stem}_{datetime.now():%Y%m%d-%H%M%S}{file.suffix}"
    dst.parent.mkdir(exist_ok=True)
    shutil.move(str(file), dst)
    return dst


def carry_over_edits(previous: Path, df: pd.DataFrame) -> int:
    """Re-apply expert decisions from a previous correction sheet, matched by sentence text.

    An edit that moved a sentence into the group named X is re-expressed as 'join the group that the
    previous members of X now belong to', because proposal names can change between versions."""
    prev = pd.read_excel(previous, sheet_name="Sentences", keep_default_na=False)
    prev["final_event"] = prev["final_event"].astype(str).str.strip()
    edited = prev[(prev.final_event != prev.proposed_event) | (prev.notes.astype(str).str.strip() != "")]
    new_prop = dict(zip(df.sentence, df.proposed_event))
    pos = {s: i for i, s in enumerate(df.sentence)}
    n = 0
    for _, r in edited.iterrows():
        if r.sentence not in pos:
            continue
        target = r.final_event
        mates = prev[(prev.proposed_event == target) & (prev.sentence != r.sentence)].sentence
        now = Counter(new_prop[s] for s in mates if s in new_prop)
        df.at[pos[r.sentence], "final_event"] = now.most_common(1)[0][0] if now else target
        note = str(r.notes).strip()
        if "carried over from" not in note:  # keep the original provenance on repeated regenerations
            note = (note + "; " if note else "") + f"carried over from {previous.name} (was {target})"
        df.at[pos[r.sentence], "notes"] = note
        n += 1
    return n


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
