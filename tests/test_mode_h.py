#!/usr/bin/env python3
"""Mode H (user 14/11: no hand-written industry function at inference): with ZC_HIDE_API=industry every phase drops the
toolkit names of src/industry.py — rule-card hints / checks (lrl.Library), compiled hints carried into links (link.card_index),
the judge's rulebook block, the names W1 may bind (lrl_gen.api_names) and the solver's API spec. Without it nothing changes."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), "src"); sys.path.insert(0, SRC)


def main():
    os.environ.pop("ZC_HIDE_API", None)
    import lrl, industry, api_spec, lrl_gen, judge
    names = set(industry.INDUSTRY_NAMES)
    card = {"kind": "definition", "content": "Số xe: số xe khác nhau", "code_hint": "n_vehicles(p, min_paid=2)",
            "check": "'n_vehicles' not in code", "trigger": {"words": ["so xe"]}}
    assert lrl.Library([dict(card)]).cards[0]["code_hint"] == "n_vehicles(p, min_paid=2)", "no hide: unchanged"
    assert any(n in lrl_gen.api_names() for n in names), "no hide: industry names valid"
    os.environ["ZC_HIDE_API"] = "industry"
    try:
        c = lrl.Library([dict(card)]).cards[0]
        assert c["code_hint"] is None and c.get("hint_hidden") and c["check"] is None, c
        assert c["content"] == card["content"], "the model-written definition stays"
        keep = lrl.strip_hidden({"code_hint": "revenue(p)"})
        assert keep["code_hint"] == "revenue(p)", "core toolkit hints stay"
        assert not (names & lrl_gen.api_names()), "W1 cannot bind a hidden name"
        blk = judge.rules_block([dict(card)])
        assert "n_vehicles" not in blk and "Số xe" in blk, blk
        spec = {e["name"] for e in api_spec.build_api_spec(hide=("industry",))}
        assert not (names & spec), "solver spec hides industry"
        shipped = lrl.Library.load(os.path.join(os.path.dirname(SRC), "Process", "archive_public_refs", "compile_public_b.jsonl"))
        assert not any(lrl.mentions(x.get("code_hint"), lrl.hidden_names()) for x in shipped.cards), "shipped public cards leak"
    finally:
        os.environ.pop("ZC_HIDE_API", None)
    print("test_mode_h PASS")


if __name__ == "__main__":
    main()
