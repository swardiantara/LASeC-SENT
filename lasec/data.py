"""Loading and preparing the catalogue (training) and FPV (held-out test) data."""
import csv
import glob
import os
import re
from pathlib import Path

import pandas as pd
import yaml

from .segment import split_messages, split_sentences

FPV_TEXT_COLUMNS = (("APP.tip", "tip"), ("APP.warning", "warning"))
FPV_TIME_COLUMNS = ("CUSTOM.date [local]", "CUSTOM.updateTime [local]")


# ----------------------------------------------------------------------------- catalogue
def _msg_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_droptc_sentences(train_xlsx, test_xlsx, old_catalogue_csv=None) -> pd.DataFrame:
    """Sentence-level catalogue from the manually segmented DroPTC dataset (Silalahi et al., 2026).

    One row per (message, sentence) occurrence. Identical rows caused by messages listed twice are
    removed. DroPTC's own train/test split and problem-type label are kept as context only: for LASeC
    both DroPTC splits form the training catalogue; the held-out test set is the FPV flights.
    If the original message catalogue is given, each message is linked to its old message-level event ID.
    """
    parts = []
    for split, f in (("train", train_xlsx), ("test", test_xlsx)):
        d = pd.read_excel(f)
        d["droptc_split"] = split
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["message"] = d["message"].astype(str).str.strip()
    d["sentence"] = d["sentence"].astype(str).str.strip()
    d = d.drop_duplicates(["message", "sentence", "problem_type"]).reset_index(drop=True)
    ids = {m: i + 1 for i, m in enumerate(dict.fromkeys(d["message"]))}
    d["line_id"] = d["message"].map(ids)
    d["sent_pos"] = d.groupby("line_id").cumcount()
    d["old_event_id"] = ""
    if old_catalogue_csv is not None and os.path.exists(old_catalogue_csv):
        old = pd.read_csv(old_catalogue_csv)
        old_map = {_msg_key(c): e for c, e in zip(old["Content"], old["EventId"])}
        d["old_event_id"] = d["message"].map(lambda m: old_map.get(_msg_key(m), ""))
    return d[["line_id", "message", "sent_pos", "sentence", "problem_type", "droptc_split", "old_event_id"]]


def segmentation_agreement(occ: pd.DataFrame, sep=";", language="en") -> dict:
    """How often ';' + PySBD reproduces the manual sentence boundaries of a message.
    Messages whose manual sentences do not cover the full message text are skipped."""
    norm = lambda xs: [re.sub(r"\W", "", x.lower()) for x in xs]
    agree, n, skipped = 0, 0, 0
    for m, g in occ.groupby("message", sort=False):
        manual = norm(g.sort_values("sent_pos").sentence.tolist())
        if "".join(manual) != re.sub(r"\W", "", m.lower()):
            skipped += 1
            continue
        auto = [s for a in split_messages(m, sep) for s in split_sentences(a, language)]
        agree += norm(auto) == manual
        n += 1
    return dict(messages=n, skipped_incomplete=skipped, pysbd_matches_manual=agree,
                share=agree / n if n else float("nan"))


def segment_catalogue(catalogue_csv: str | os.PathLike, sep=";", language="en") -> pd.DataFrame:
    """One row per (message, sentence) occurrence of the message catalogue."""
    df = pd.read_csv(catalogue_csv)
    rows = []
    for _, r in df.iterrows():
        for mi, msg in enumerate(split_messages(r["Content"], sep)):
            for si, s in enumerate(split_sentences(msg, language)):
                rows.append(dict(line_id=r["LineId"], message=r["Content"], msg_part=mi, sent_pos=si,
                                 sentence=s, old_event_id=r["EventId"], old_template=r["EventTemplate"]))
    return pd.DataFrame(rows)


def unique_sentences(occ: pd.DataFrame) -> pd.DataFrame:
    """Unique sentence strings with their occurrence count."""
    g = occ.groupby("sentence", sort=True)
    return pd.DataFrame({"sentence": list(g.groups), "occurrences": g.size().values})


# ----------------------------------------------------------------------------- FPV evidence
def read_fpv_flight(csv_file: str | os.PathLike) -> pd.DataFrame:
    """Extract timestamped tip/warning cells from one decrypted DJI flight record.

    The decrypted CSV starts with a 'sep=,' line, followed by the header.
    """
    with open(csv_file, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0].startswith("sep="):
        rows = rows[1:]
    header, body = rows[0], rows[1:]
    idx = {c: header.index(c) for c in (*FPV_TIME_COLUMNS, *(c for c, _ in FPV_TEXT_COLUMNS))}
    out = []
    for row_i, r in enumerate(body):
        for col, kind in FPV_TEXT_COLUMNS:
            v = r[idx[col]] if len(r) > idx[col] else ""
            if v.strip():
                out.append(dict(row=row_i, date=r[idx[FPV_TIME_COLUMNS[0]]], time=r[idx[FPV_TIME_COLUMNS[1]]],
                                kind=kind, cell=v))
    return pd.DataFrame(out)


def segment_fpv(decrypted_dir: str | os.PathLike, sep=";", language="en") -> pd.DataFrame:
    """One row per sentence occurrence across all flights (flight numbered by sorted file name)."""
    files = sorted(glob.glob(os.path.join(decrypted_dir, "*.csv")))
    rows = []
    for flight, f in enumerate(files, 1):
        cells = read_fpv_flight(f)
        for cell_i, c in cells.iterrows():
            for mi, msg in enumerate(split_messages(c["cell"], sep)):
                for si, s in enumerate(split_sentences(msg, language)):
                    rows.append(dict(flight=flight, file=Path(f).name, row=c["row"], date=c["date"], time=c["time"],
                                     kind=c["kind"], cell_id=f"{flight}-{cell_i}", cell=c["cell"],
                                     msg_part=mi, message=msg, sent_pos=si, sentence=s))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- labels
def load_labels(labels_csv: str | os.PathLike, column: str) -> dict:
    df = pd.read_csv(labels_csv)
    df = df[df[column].notna() & (df[column].astype(str).str.strip() != "")]
    return dict(zip(df["sentence"], df[column].astype(str).str.strip()))


def attach_labels(units: pd.DataFrame, labels: dict, what: str) -> pd.DataFrame:
    missing = sorted(set(units["sentence"]) - set(labels))
    if missing:
        raise ValueError(f"{len(missing)} {what} sentences have no label, e.g. {missing[:5]}")
    units = units.copy()
    units["event"] = units["sentence"].map(labels)
    return units


def load_key_events(yaml_file: str | os.PathLike) -> dict:
    with open(yaml_file, encoding="utf-8") as f:
        return {int(k): v for k, v in yaml.safe_load(f)["flights"].items()}
