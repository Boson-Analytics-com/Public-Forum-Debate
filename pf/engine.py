"""
Step 13: the debate engine.

Real side-vs-side debate: you pick a side, the tool takes the other, both go
through the full 11-step PF order (see round.py's DEBATE_ORDER), and the
finished debate is handed to the judge.

This module does not build its own model client. It reuses round.py's
client_or_die(), which returns a BedrockClientAdapter backed by the AWS
credential chain (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, an AWS profile, or
an IAM role). That one client is threaded through debate.py, round.py and
judge.py for the whole debate, so every stage -- and the judge call at the
end -- goes through the same Bedrock Converse API call. Nothing in this
pipeline touches ANTHROPIC_API_KEY.

Usage:
    python engine.py debate --corpus ./corpus --human-side pro
    python engine.py debate --corpus ./corpus --human-side con --exchanges 4
"""

import argparse
import json
from pathlib import Path

from pf import judge as jdg
from pf import round as rnd


def run_debate(corpus: Path, cfg: dict, human_side: str, client, mock: bool,
              crossfire_exchanges: int | None = None, resume: bool = False):
    """The real side-vs-side debate: human picks a side, tool takes the
    other, and both go through the full 11-step fixed PF order --
    constructive, crossfire, rebuttal, crossfire, summary, grand crossfire,
    final focus -- with a human turn at every stage that belongs to the
    human's side. This is engine.py's entry point into round.py's
    debate-mode functions (debate_start / debate_run)."""
    sp = rnd.debate_state_path(corpus)
    if sp.exists() and not resume:
        print(f"[resuming existing debate at {sp}; pass --resume to silence this note, "
              f"or delete the file to start over]")
    if not sp.exists():
        rnd.debate_start(corpus, cfg, human_side, client, mock)

    path = rnd.debate_run(corpus, cfg, client, mock, crossfire_exchanges=crossfire_exchanges)

    print(f"\n{'=' * 72}\n= JUDGE\n{'=' * 72}")
    jdg.score(path, client, mock)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["debate"])
    ap.add_argument("--corpus", type=Path, default=Path("./corpus"))
    ap.add_argument("--config", type=Path, default=Path("config/resolution.json"))
    ap.add_argument("--human-side", choices=["pro", "con"], default=None,
                    help="which side you argue -- the tool automatically takes the other")
    ap.add_argument("--exchanges", type=int, default=None,
                    help="max exchanges per crossfire stage "
                         "(default: the value saved at debate start, else 6)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing debate_state.json without the resume note")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    if not args.human_side:
        raise SystemExit("debate needs --human-side pro|con")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    client = rnd.client_or_die(args.mock)
    run_debate(args.corpus, cfg, args.human_side, client, args.mock,
              crossfire_exchanges=args.exchanges, resume=args.resume)


if __name__ == "__main__":
    main()
