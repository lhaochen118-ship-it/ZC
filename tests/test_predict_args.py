#!/usr/bin/env python3
"""src/predict_flow.py builds the agent command from DEFAULTS + --config: a flag the config sets must not be passed on the
command line too (an explicit flag beats the config: '--prompt compact' used to override the final config's full prompt).
Checks: with configs/flow_final.json the parsed agent flags equal the config's values; without a config, the DEFAULTS apply.
Usage: test_predict_args.py"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import predict_flow, agent


def main():
    cfg = os.path.join(ROOT, "configs", "flow_final.json")
    c = json.load(open(cfg))
    argv = ["split", "out", "--model", "m"] + predict_flow.defaults_for(cfg) + ["--config", cfg]
    a = agent.parse_with_config(agent.build_parser(), argv)
    bad = {k: (getattr(a, k), v) for k, v in c.items() if not k.startswith("_") and getattr(a, k) != v}
    assert not bad, f"config values overridden: {bad}"
    a0 = agent.parse_with_config(agent.build_parser(), ["split", "out", "--model", "m"] + predict_flow.defaults_for(""))
    assert a0.prompt == "compact" and a0.max_tokens == 512 and a0.flow == "hybrid", (a0.prompt, a0.max_tokens)
    print(f"flow_final: {len(c) - 1} config values kept (prompt {a.prompt}, variants {a.variants}); defaults without a config OK")
    print("test_predict_args PASS")


if __name__ == "__main__":
    main()
