"""Score the actual native HF ACT with its separately stored local residual head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from real2sim.artifacts import sha256_file
from real2sim.policy.local_hf_head import LocalHeadACT
from real2sim.serialization import write_json

if __package__:
    from scripts import evaluate_hf_act_20261027 as so
    from scripts import evaluate_hf_act_panda_20261028 as panda
else:
    import evaluate_hf_act_20261027 as so
    import evaluate_hf_act_panda_20261028 as panda


def run(robot, head_dir, out, seeds, snapshot=so.SNAPSHOT, *, video=False):
    metadata = json.loads((head_dir / "head.json").read_text())
    before = {str(path): sha256_file(path) for path in (
        head_dir / "head.pt", head_dir / "head.json", Path(__file__),
        Path("src/real2sim/policy/local_hf_head.py"))}

    def load(path):
        return LocalHeadACT(path, head_dir)

    runner = so if robot == "so101" else panda
    with patch.object(runner, "NativeLeRobotACT", load):
        report = runner.run(snapshot, out, seeds, execute_steps=10, video=video)
    after = {path: sha256_file(path) for path in before}
    if before != after:
        raise RuntimeError("Local adapter source changed during evaluation")
    identifier = hashlib.sha256((metadata["base_checkpoint_sha256"] + metadata["head_sha256"]).encode()).hexdigest()
    for row in report["trials"]:
        row.update(experiment_id="NATIVE_HF_WITH_LOCAL_RESIDUAL_HEAD_20261028",
                   controller_source="NATIVE_HF_ACT_WITH_LOCAL_RESIDUAL_ACTION_HEAD",
                   local_head_sha256=metadata["head_sha256"], effective_policy_identifier=identifier)
    report.update(schema="native_hf_local_head_evaluation_20261028_v1", robot=robot,
        actual_hf_visual_encoder_and_transformer_used=True, semantic_act_used=False,
        local_head=metadata, local_head_directory=str(head_dir),
        effective_policy_identifier=identifier, no_runtime_teacher_or_corrective_object_goals=True,
        lighting_conditions_changed=False, head_source_sha256_before=before,
        head_source_sha256_after=after, local_only=True, hf_upload_performed=False)
    write_json(out / "report.json", report, atomic=True)
    (out / "executed_local_head_wrapper.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"robot": robot, "head": str(head_dir), "summary": report["summary"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("so101", "panda"), required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[300, 301, 302, 303, 304])
    parser.add_argument("--snapshot", type=Path, default=so.SNAPSHOT)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    run(args.robot, args.head, args.out, args.seeds, args.snapshot, video=args.video)
