#!/usr/bin/env python3
import os, time, glob


def left_s():
    dl = float(os.environ.get("ZC_DEADLINE") or 0)
    return (dl - time.time()) if dl else None


def callback(TrainerCallback, reserve_s, stats):
    class DeadlineStop(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            left = left_s()
            if left is not None and left < reserve_s and state.global_step < state.max_steps:
                control.should_save = True; control.should_training_stop = True
                stats["stopped_early"] = {"step": state.global_step, "max_steps": state.max_steps}
                print(f"[deadline] {left / 60:.0f} min before the session deadline: checkpoint at step {state.global_step}"
                      f"/{state.max_steps} and stop (resume it in a new kernel)", flush=True)
            return control
    return DeadlineStop()


def latest_checkpoint(path):
    if not path or not os.path.exists(path): return None
    if os.path.isfile(os.path.join(path, "trainer_state.json")): return path
    c = glob.glob(os.path.join(path, "**", "checkpoint-*", "trainer_state.json"), recursive=True)
    step = lambda p: int(os.path.basename(os.path.dirname(p)).split("-")[-1])
    return os.path.dirname(max(c, key=step)) if c else None
