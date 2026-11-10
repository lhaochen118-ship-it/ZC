#!/usr/bin/env python3
"""Kaggle stops a commit session at 12 h and its outputs are then lost (user 10/11). The kernel runner
(automation/make_kernel.py) gives every job ZC_DEADLINE (epoch seconds, 12 h minus a margin after the session started); a
training run that would go past it saves a resumable checkpoint (LoRA weights + optimizer + scheduler + RNG + step, the HF
Trainer format) and stops, its job marks the branch 'partial' and skips the evaluation, and the next kernel - mounting this
kernel's output - resumes from the checkpoint (`--resume`), so a training of any length runs as a chain of sessions.

    callback(TrainerCallback, reserve_s, stats) -> a callback that checkpoints + stops `reserve_s` before ZC_DEADLINE
    latest_checkpoint(path) -> the newest checkpoint-N directory under path (or path itself), else None
    left_s() -> seconds before ZC_DEADLINE (None without a deadline)"""
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
