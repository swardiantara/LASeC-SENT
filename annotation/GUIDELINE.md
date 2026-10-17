# Annotation guideline: event identity in drone flight-log sentences

Version 2.0 (October 2026). Applies to the training catalogue (`data/raw/catalogue`, DroPTC sentence split) and
the held-out DJI FPV test set (`data/raw/fpv`). Both sets are labelled under this same guideline.

## 1. Purpose

The labels define the ground truth for **forensic triage**: from a flight log, identify the **unique events**
that occurred, so that an investigator reviews one representative per event instead of every log line.

## 2. Unit of annotation

One sentence, labelled independently. The source message is shown only as context.
- *Training catalogue:* sentence boundaries come from the **manual sentence split of the DroPTC dataset**
  (`train_sentence.xlsx` + `test_sentence.xlsx`; both DroPTC splits form the LASeC catalogue). Vendor catalogue
  messages are inconsistently punctuated, so automatic boundaries are not used for the training labels.
- *FPV test set:* boundaries come from the LASeC pre-processing used at inference time: split each decrypted log
  cell into app messages at `;`, then into sentences with PySBD. The FPV logs are consistently punctuated, and the
  expert checked every resulting sentence during labelling.

## 3. Criteria

**C1: Events are what we group.** An *event sentence* reports something that happened or holds on the
aircraft, its link or its surroundings: a state change, an action of the system, or a condition. Examples:
`Compass error.`, `RC signal lost.`, `Landing gear lowered.`, `Aircraft returning to home.`,
`Max altitude set to 30 m.` The following are **not events**:

| Type | What it is | Examples |
|---|---|---|
| instruction | operating guidance for the pilot | `Fly with caution.`, `Press C2 to exit`, `Tap the screen to focus.` |
| mitigation | a remedy recommended for a problem | `Calibrate IMU`, `Land as soon as possible`, `Contact DJI Support` |
| possible | a hedged or predicted consequence, not an occurrence | `Signal interruption may be more likely.`, `Flight performance will be affected.`, `flight time is shortened.` |
| info | static information or bookkeeping | `Minimum RTH Altitude is 30m.`, `Data Recorder File Index is 12.` |

"The same underlying situation or condition" means **the same underlying event**.

**C2: Sentences with several phrases.** Label by the **event part**:
- event + non-event part (instruction, consequence, possibility) → the label of the event alone.
  `Aircraft is in Attitude mode and hovering may be unstable.` = `Aircraft in Attitude mode.`;
  `Compass error, calibration required` = `Compass error.`
- two or more events → a **compound** label of its own, distinct from each single-event sentence even when they
  overlap. `Ambient Light too Weak, Forward obstacle sensing is unavailable.` ≠ `Ambient light too low.`;
  `Aircraft exceeded distance limit and has exited Hyperlapse mode` ≠ `Exited Hyperlapse mode`.
- a cause or consequence stated as a fact counts as an event; a hedged one ("may", "will be affected") does not.

**C3: Merge semantically similar sentences with the same underlying event,** regardless of:
- wording, paraphrase and word order (`Aircraft returning.` = `Returning to home.` = `RTH in progress.`);
- parameter values (`Auto RTH in 3 s.` = `Auto RTH in 9 s.`; `Battery Temperature Below 15°C` = `Battery temperature too low.`);
- severity or intensity of the same condition (`Manned aircraft detected nearby` = `Manned aircraft too close`);
- the sensor or direction that reports an environmental condition (`Forward ambient light too low.` = `Ambient light too low.`);
- capitalisation, punctuation, typos, `Warning:` / `Please` / `Tip:` prefixes.

Keep apart sentences whose underlying events differ:
- different components, axes, links or functions (gimbal pitch vs yaw limit; RC signal vs image transmission vs downlink);
- opposite or different states (on/off, locked/unlocked, weak/lost, lost/restored, approaching/reached a limit);
- a different trigger, variant or phase of a function (`Failsafe RTH.`, `Low battery RTH.`, `RTH ascending.`, `Auto RTH canceled.`);
- a distinct threshold state with its own system response (`Low battery.` vs `Critical low battery.`, which triggers landing);
- flight-mode-change notifications (`Flight mode changed to X`, `Switched to X-mode`) vs reports of the activity
  itself (`Motors starting.`, `Auto landing in progress.`).

**When unsure, keep apart** and write `UNSURE` in the notes. Over-merging can hide an event from the investigator.

## 4. Labelling non-event sentences

Non-event sentences are still grouped by meaning, so they can be included in or excluded from training and
evaluation later without relabelling: `INSTR_` (instructions and mitigations, grouped by the requested action),
`POSS_` (possible events), `INFO_` (information). The sentence type is recorded in its own column.

## 5. Worked examples (expert decisions)

| Sentences | Decision | Criterion |
|---|---|---|
| `Aircraft in Attitude mode.` / `Aircraft is in Attitude mode and hovering may be unstable.` / `Aircraft is in Attitude mode, so that it will not hover.` | one event `AIRCRAFT_ATTITUDE_MODE` | C2 (consequence dropped), C3 |
| `aircraft braking distance is extended` / `Aircraft braking distance increased and flight time decreased.` | one event `AIRCRAFT_BRAKING_DISTANCE_EXTENDED` | C2 (flight time is a consequence) |
| `Your aircraft is at the boundary of an Altitude Zone.` vs `Your aircraft is flying in an Altitude Zone (nnn).` | two events | C3 (different state) |
| `Ambient light too low.` / `Forward …` / `Downward ambient light too low.` | one event `AMBIENT_LIGHT_LOW` | C3 (reporting direction) |
| `Aircraft returning to home.` / `Aircraft returning.` / `Returning to home.` / `RTH in progress.` / `RTH.` | one event `RTH_ACTIVE` | C3 |
| `Failsafe RTH.` vs `Low battery RTH.` vs `Original route RTH.` vs `RTH ascending.` | four events | C3 (trigger, variant, phase) |
| `Low battery.` vs `Low battery RTH.` | two events | C2 (two events in one sentence) |
| `RC signal weak.` vs `RC signal lost.` | two events | C3 (different state) |
| `Motors starting.` vs `Flight mode changed to Starting Motors.` | two events | C3 (mode notification vs activity) |
| `Set Return to Home (RTH) altitude to 100 m (328 ft).` / `RTH altitude adjusted to 100m.` | one event `RTH_ALT_SET` | C3 |
| `Adjust antennas and make sure …` / `Avoid blocking antennas and adjust antenna orientation.` | non-event `INSTR_ADJUST_ANTENNA` | C1, Section 4 |
| `QuickShot has ended` and `Target lost.` + `QuickShots stopped` | `QuickShots stopped` = `QuickShot has ended`; `Target lost.` is its own event | C3 |

## 6. Event ID conventions

- `UPPER_SNAKE_CASE`, describing subject + state (+ target): `GPS_SIGNAL_WEAK`, `GIMBAL_PITCH_LIMIT`.
- Prefixes: `MODE_` flight-mode changes; `INSTR_` / `POSS_` / `INFO_` non-events; settings end in `_SET`.
- **Reuse the FPV IDs** whenever a catalogue sentence describes the same event, so the train and test label
  spaces stay consistent.
- IDs are names only. Evaluation uses which sentences share an ID.

## 7. Process

1. **Proposal.** Claude labelled all 783 catalogue sentences under this guideline
   (`annotation/catalogue_labels_v2_proposed.csv`, with a rationale per sentence). Decisions where the criteria allow two readings are marked `check`.
2. **Expert review.** The expert edits `final_type` / `final_label` in `annotation/catalogue_review_v2.xlsx`
   (built by `scripts/build_review_sheet.py`). The `changed` column records the correction rate, which is reported in the paper.
3. **Freeze.** `scripts/import_annotations.py` validates and exports the labels to `data/labels/` before any model is trained or evaluated.

## 8. Quality checklist (before freezing)

- [ ] Every sentence has a type and a label; IDs are UPPER_SNAKE_CASE.
- [ ] No label mixes event and non-event types.
- [ ] Every compound label contains two or more events; every event + instruction sentence carries the event's label.
- [ ] Sentences differing only by numbers share a label.
- [ ] Opposite states never share a label.
- [ ] Every `needs_check` row has been decided.
- [ ] FPV IDs reused where applicable; FPV sentence types reviewed (sheet *FPV events*).
