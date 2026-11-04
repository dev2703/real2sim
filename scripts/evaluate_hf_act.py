"""Evaluate the latest pinned native Hugging Face ACT on SO-101 or Panda.

Keeps the original weights and saved processors, and exposes the same source
scene for both embodiments. Native raw-policy success remains separate from
any geometric teacher or assisted controller.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from scripts import evaluate_hf_act_20261027 as so101
else:
    import evaluate_hf_act_20261027 as so101


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("so101", "panda"), required=True)
    parser.add_argument("--snapshot", type=Path, default=so101.SNAPSHOT)
    parser.add_argument("--source-scene", type=Path, default=Path(
        "configs/v2/measured_registration_20261028/sim_so101_positive_all_scalars.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[300, 301, 302, 303, 304])
    parser.add_argument("--execute-steps", type=int)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.robot == "so101":
        return so101.run(args.snapshot, args.out, args.seeds, args.execute_steps,
                         video=args.video, source_scene=args.source_scene)
    if __package__:
        from scripts import evaluate_hf_act_panda_20261028 as panda
    else:
        import evaluate_hf_act_panda_20261028 as panda

    return panda.run(args.snapshot, args.out, args.seeds,
                     execute_steps=args.execute_steps, video=args.video,
                     source_scene=args.source_scene)


if __name__ == "__main__":
    main()
