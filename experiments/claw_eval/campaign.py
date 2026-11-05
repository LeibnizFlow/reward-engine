"""Runtime-neutral Claw-Eval entry; the historical entry remains compatible."""

from experiments.claw_eval.opencode.campaign import main, prepare, report, serve

__all__ = ["main", "prepare", "report", "serve"]


if __name__ == "__main__":
    main()
