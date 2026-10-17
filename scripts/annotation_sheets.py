"""Excel writers for the annotation correction sheet and the blind second-annotator sheets."""
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

F = "Arial"
HDR = PatternFill("solid", fgColor="1F3864")
EDIT = PatternFill("solid", fgColor="FFF2CC")
BOX = Border(*(Side(style="thin", color="BFBFBF"),) * 4)
WRAP = Alignment(wrap_text=True, vertical="top")


def _header(ws, cols, widths, row=1):
    for j, (c, w) in enumerate(zip(cols, widths), 1):
        cell = ws.cell(row, j, c)
        cell.fill, cell.font, cell.border = HDR, Font(name=F, bold=True, color="FFFFFF", size=10), BOX
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = w


def _cell(ws, r, c, v, edit=False, bold=False):
    cell = ws.cell(r, c, v)
    cell.font, cell.alignment, cell.border = Font(name=F, size=10, bold=bold), WRAP, BOX
    if edit:
        cell.fill = EDIT
    return cell


def _notes(ws, lines, width=120):
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = width
    for i, (a, b) in enumerate(lines, 1):
        ws.cell(i, 1, a).font = Font(name=F, size=11 if i == 1 else 10, bold=True, color="1F3864" if i == 1 else None)
        c = ws.cell(i, 2, b)
        c.font, c.alignment = Font(name=F, size=10), WRAP


def write_correction_sheet(df, fpv, fpv_col, out):
    wb = Workbook()
    ws = wb.active
    ws.title = "How to use"
    _notes(ws, [
        ("Catalogue sentence labelling", "Expert correction sheet. Read annotation/GUIDELINE.md first."),
        ("Goal", "Give every catalogue sentence an event ID such that two sentences share an ID iff they describe "
                 "the same underlying situation/condition (GUIDELINE.md, Sections 2-4)."),
        ("Where to edit", "Sheet 'Sentences', yellow columns only: 'final_event' (pre-filled with the proposal) and 'notes'."),
        ("Step 1", "Work top to bottom; rows are sorted by family, then proposed event, so each proposed group is contiguous."),
        ("Step 2", "Rows with 'review_flags' need a decision; unflagged rows are high-precision proposals but still check them."),
        ("Step 3", "To MERGE two groups: give both the same final_event. To SPLIT: give the deviating rows a new ID."),
        ("Step 4", "Reuse FPV event IDs (sheet 'FPV events') whenever a catalogue sentence describes the same event."),
        ("Step 5", "Unsure? Keep apart and write UNSURE in notes. Segmentation fragment? Label it and write SEG_ERROR in notes."),
        ("Step 6", "Check sheet 'Events': live member counts per final_event, to spot accidental typos in IDs (count 1 where you expected more)."),
        ("Column 'changed'", "Formula: shows 'yes' when final_event differs from the proposal (used to report the correction rate)."),
        ("Column 'basis'", "FPV link = same as an expert-labelled FPV sentence; exact-norm = identical after masking numbers/case/prefixes; "
                           "old-label = both were whole messages with the same old message-level event; embedding = pre-trained "
                           "embedding similarity (least reliable); singleton = no proposal partner."),
        ("Column 'nearest_*'", "The closest other proposed event and one of its sentences: the most likely missed merge."),
        ("Column 'family_hint'", "Inferred sentence family (event type): status, instruction, mode_change, setting, record. "
                                 "Different families never share an event (GUIDELINE.md, Step 1). Correct it in notes if wrong."),
        ("Column 'problem_type'", "DroPTC problem type of the sentence (context). Members of one event normally share it; "
                                  "proposals never merge different problem types by embedding similarity alone. "
                                  "A few sentences have a context-dependent type (e.g. 'Unable to take off.'), shown joined by '/'."),
        ("Carried-over rows", "Decisions made in an earlier version of this sheet are re-applied by sentence text and marked "
                              "'carried over' in notes. The earlier sheet is kept in annotation/archive/."),
        ("Sentences", "Sentence boundaries come from the manual DroPTC split (annotation/train_sentence.xlsx, test_sentence.xlsx). "
                      "Both DroPTC splits form the LASeC training catalogue."),
        ("Example", "Row 'Compass Error.' proposed COMPASS_ERROR (basis FPV link): keep. Row 'Exit P-GPS Mode' flagged "
                    "'family uncertain': decide whether it is an instruction or a status and set e.g. MODE_EXIT_PGPS."),
        ("Second annotator", "Do NOT share this file with annotator 2. They label catalogue_iaa_annotator2.xlsx and "
                             "fpv_iaa_annotator2_blind.xlsx, which contain no proposals or expert labels."),
    ])

    cols = ["sent_id", "sentence", "occurrences", "family_hint", "problem_type", "proposed_event", "basis", "group_size",
            "review_flags", "nearest_other_event", "nearest_distance", "nearest_example", "source_message",
            "old_message_events", "final_event", "changed", "notes", "iaa_subset"]
    widths = [8, 48, 8, 12, 16, 28, 11, 7, 26, 26, 9, 36, 48, 14, 30, 8, 30, 7]
    L = {c: get_column_letter(i) for i, c in enumerate(cols, 1)}
    ss = wb.create_sheet("Sentences")
    _header(ss, cols, widths)
    ss.cell(1, cols.index("problem_type") + 1).value = "problem_type (DroPTC)"
    for r, row in enumerate(df.itertuples(index=False), 2):
        d = row._asdict()
        for c, col in enumerate(cols, 1):
            if col == "changed":
                _cell(ss, r, c, f'=IF(TRIM({L["final_event"]}{r})={L["proposed_event"]}{r},"","yes")')
            elif col in ("final_event", "notes"):
                _cell(ss, r, c, d.get(col, d["proposed_event"] if col == "final_event" else ""), edit=True)
            else:
                _cell(ss, r, c, d[col])
    n = len(df) + 1
    ss.freeze_panes = "C2"
    ss.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{n}"
    fl, chg = L["review_flags"], L["changed"]
    ss.conditional_formatting.add(f"{fl}2:{fl}{n}", FormulaRule(formula=[f'LEN(${fl}2)>0'], fill=PatternFill("solid", fgColor="F8CBAD")))
    ss.conditional_formatting.add(f"{chg}2:{chg}{n}", FormulaRule(formula=[f'${chg}2="yes"'], font=Font(name=F, bold=True, color="C00000")))

    ev = wb.create_sheet("Events")
    _header(ev, ["proposed_event", "family_hint", "proposed_members", "members (proposal)", "rows now labelled with this ID"],
            [30, 12, 10, 90, 16])
    g = df.groupby("proposed_event", sort=False)
    for r, (e, sub) in enumerate(g, 2):
        _cell(ev, r, 1, e)
        _cell(ev, r, 2, sub.family_hint.mode()[0])
        _cell(ev, r, 3, len(sub))
        _cell(ev, r, 4, " | ".join(sub.sentence))
        fe_col = L["final_event"]
        _cell(ev, r, 5, f"=COUNTIF(Sentences!${fe_col}$2:${fe_col}${n},A{r})")
    ev.freeze_panes = "B2"
    ev.auto_filter.ref = f"A1:E{len(g) + 1}"

    fe = wb.create_sheet("FPV events")
    _header(fe, ["FPV event ID (expert)", "FPV sentences"], [30, 110])
    for r, (e, sub) in enumerate(fpv.groupby(fpv_col), 2):
        _cell(fe, r, 1, e)
        _cell(fe, r, 2, " | ".join(sub.sentence))
    for sh in wb.worksheets:
        sh.sheet_view.zoomScale = 90
    wb.save(out)


def write_blind_sheet(df, out, what):
    wb = Workbook()
    ws = wb.active
    ws.title = "How to use"
    _notes(ws, [
        (f"{what} sentence labelling (annotator 2)", "Blind labelling for inter-annotator agreement. Read annotation/GUIDELINE.md first."),
        ("Task", "Give each sentence an event ID. Two sentences get the same ID iff they describe the same underlying "
                 "situation/condition. ID names are free (UPPER_SNAKE_CASE); only which sentences share an ID matters."),
        ("Where to edit", "Sheet 'Label', yellow columns 'event' and 'notes'. Do not consult other label files."),
        ("Example", "'Auto RTH in 3 s.' and 'Auto RTH in 7 s.' -> both AUTO_RTH_COUNTDOWN (parameter values are ignored)."),
        ("Unsure", "Keep sentences apart and write UNSURE in notes."),
    ], width=100)
    has_ctx = "source_message" in df and df["source_message"].astype(str).str.len().gt(0).any()
    cols = ["sent_id", "sentence"] + (["source_message (context)"] if has_ctx else []) + ["event", "notes"]
    ls = wb.create_sheet("Label")
    _header(ls, cols, [8, 60] + ([60] if has_ctx else []) + [30, 24])
    for r, row in enumerate(df.itertuples(index=False), 2):
        vals = [row.sent_id, row.sentence] + ([row.source_message] if has_ctx else [])
        for c, v in enumerate(vals, 1):
            _cell(ls, r, c, v)
        _cell(ls, r, len(vals) + 1, "", edit=True)
        _cell(ls, r, len(vals) + 2, "", edit=True)
    ls.freeze_panes = "C2"
    wb.save(out)
