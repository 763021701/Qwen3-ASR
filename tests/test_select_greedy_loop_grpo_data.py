from argparse import Namespace

from tools.select_greedy_loop_grpo_data import decide_weight, loop_penalties, select_records

ARGS = Namespace(
    weight_hard_loop=4,
    weight_hard=2,
    weight_easy_loop=2,
    weight_anchored_loop=2,
    loop_pen_min=0.2,
    clean_cer_max=0.30,
)

REF = "specimen labelled left pelvic lymph nodes submitted for examination"
LOOPED = REF + " for examination for examination for examination"


def test_decide_weight_truth_table():
    assert decide_weight("hard", True, True, ARGS) == 4
    assert decide_weight("hard", False, True, ARGS) == 2
    assert decide_weight("easy", True, True, ARGS) == 2
    assert decide_weight("easy", False, True, ARGS) == 0
    assert decide_weight("weak", True, True, ARGS) == 2
    assert decide_weight("weak", True, False, ARGS) == 0
    assert decide_weight("suspect", True, True, ARGS) == 2
    assert decide_weight("suspect", False, True, ARGS) == 0
    # suspect looping with no clean rollout: label risk band, never trained on
    assert decide_weight("suspect", True, False, ARGS) == 0
    assert decide_weight("unknown_category", True, True, ARGS) == 0


def test_loop_penalties_flag_only_actual_repetition():
    pens = loop_penalties(REF, [REF, LOOPED])
    assert pens[0] == 0.0
    assert pens[1] >= ARGS.loop_pen_min


def test_select_records_flags_loop_active_easy_clip():
    record = {
        "miner_source_index": 0,
        "audio": "a.wav",
        "reference": REF,
        "category": "easy",
        "cers": [0.02, 0.5],
        "hypotheses": [REF, LOOPED],
    }
    items = select_records([record], ARGS)
    assert len(items) == 1
    assert items[0]["loop_active"] is True
    assert items[0]["clean_exists"] is True
    assert items[0]["weight"] == ARGS.weight_easy_loop
