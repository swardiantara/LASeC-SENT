"""Prompts for LLM-based event grouping (zero-shot and few-shot).

The LLM receives a numbered list of sentences and must return a partition of the IDs into
events. Zero-shot: event-identity rules only. Few-shot: the same rules plus solved
demonstrations of the same task, built from labelled CATALOGUE sentences only.
"""
import json
import random
from collections import defaultdict

SYSTEM = """You are assisting a digital forensic investigator who triages drone flight logs.
Each input line is one sentence taken from the human-readable messages (tips and warnings) that a
drone app records during a flight. Group the sentences into EVENTS so that the investigator can
review one representative per event.

Two sentences belong to the same group if and only if they describe the same underlying EVENT
(a state change, an action of the system, or a condition). Apply these rules:
1. Instructions, recommended remedies, hedged/possible consequences ("may", "will be affected") and static
   information are not events. Never group them with event sentences; group them only with sentences that
   request the same action or state the same information.
2. If a sentence contains an event and a non-event part, group it by its event part
   ("Aircraft is in Attitude mode and hovering may be unstable." = "Aircraft in Attitude mode.").
   If it contains two or more events, it forms its own group, apart from each single-event sentence.
3. Merge sentences with the same underlying event regardless of wording, word order, parameter values
   (numbers, durations, indices), severity wording ("nearby" / "too close"), or the sensor/direction reporting
   an environmental condition ("forward/downward ambient light too low").
4. Keep apart different components, axes, links or functions; opposite or different states (on/off, weak/lost,
   lost/restored, approaching/reached); a different trigger, variant or phase of a function ("Failsafe RTH",
   "RTH ascending"); and flight-mode-change notifications ("Flight mode changed to X") vs reports of the activity.
5. When unsure, keep the sentences apart.

Return every sentence ID exactly once."""

SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event": {"type": "string", "description": "short UPPER_SNAKE_CASE event name"},
                    "members": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["event", "members"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}


def numbered(sentences):
    return "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))


def build_demonstrations(sentences, events, exclude=(), n_groups=15, max_members=3, seed=42):
    """Sample a solved grouping task from labelled catalogue data.

    Half of the sampled events have several members (paraphrases/parameter variants), the rest are
    singletons, so the demonstration shows both merging and keeping apart. Sentences in `exclude`
    (e.g. those also present in the test set) are never shown.
    """
    excl = {s.lower().strip() for s in exclude}
    by_event = defaultdict(list)
    for s, e in zip(sentences, events):
        if s.lower().strip() not in excl:
            by_event[e].append(s)
    rng = random.Random(seed)
    multi = sorted(e for e, m in by_event.items() if len(m) > 1)
    single = sorted(e for e, m in by_event.items() if len(m) == 1)
    pick = rng.sample(multi, min(len(multi), n_groups // 2 + n_groups % 2))
    pick += rng.sample(single, min(len(single), n_groups - len(pick)))
    demo = [(s, e) for e in pick for s in rng.sample(sorted(by_event[e]), min(max_members, len(by_event[e])))]
    rng.shuffle(demo)
    groups = defaultdict(list)
    for i, (_, e) in enumerate(demo):
        groups[e].append(i)
    answer = {"groups": [{"event": e, "members": m} for e, m in groups.items()]}
    return [s for s, _ in demo], answer


def user_prompt(batch, demo=None, known_events=None):
    parts = []
    if demo:
        demo_sents, demo_answer = demo
        parts.append("Here is a solved example of the task.\n\nExample sentences:\n" + numbered(demo_sents)
                     + "\n\nExample answer:\n" + json.dumps(demo_answer))
    if known_events:
        parts.append("Events already identified in earlier batches of this log (name: representative). "
                     "Reuse a name exactly when a sentence belongs to that event; otherwise create a new name.\n"
                     + "\n".join(f"{e}: {r}" for e, r in known_events.items()))
    parts.append("Group these sentences into events. Answer with JSON only, using the schema "
                 '{"groups": [{"event": str, "members": [int, ...]}]}.\n\nSentences:\n' + numbered(batch))
    return "\n\n".join(parts)
