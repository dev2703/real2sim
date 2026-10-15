"""Join frozen control, full-grasp contact and real-image prediction evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np

from real2sim.policy.paired import file_sha256


def scores(pred, label, std):
    error = pred - label
    return {'n': len(label), 'normalized_l1': float(np.abs(error / std).mean()),
            'translation_error_mm': float(np.linalg.norm(error[:, :3], axis=1).mean() * 1000),
            'rotation_error_rad': float(np.linalg.norm(error[:, 3:6], axis=1).mean()),
            'gripper_mae_percentage_points': float(np.abs(error[:, 6]).mean() * 100)}


def first_persistent(values, threshold, n=3):
    for i in range(len(values) - n + 1):
        if all(v[1] > threshold for v in values[i:i+n]):
            return values[i][0]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ik', type=Path, default=Path('results/audit/ik_polishing_20261015/verified'))
    parser.add_argument('--replay', type=Path, default=Path('results/audit/recorded_grasp_control_20261015/verified'))
    parser.add_argument('--motion', type=Path, default=Path('results/audit/motion_prediction_20261015'))
    parser.add_argument('--out', type=Path, default=Path('results/audit/control_findings_20261015'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    ik = json.loads((args.ik / 'review.json').read_text())
    replay = json.loads((args.replay / 'report.json').read_text())
    motion = json.loads((args.motion / 'report.json').read_text())
    if len(ik['cases']) != 4 or any(set(v) != {'baseline', 'polished'} for v in ik['cases'].values()):
        raise ValueError('frozen-policy audit incomplete')
    if len(replay['episodes']) != 3:
        raise ValueError('grasp audit incomplete')
    if file_sha256(args.motion / 'predictions.npz') != motion['predictions_sha256']:
        raise ValueError('prediction checksum mismatch')
    preds = np.load(args.motion / 'predictions.npz', allow_pickle=False)
    std = np.array(json.loads(Path('.cache/derived_cartesian/stats.json').read_text())['action']['std'])
    names = ('parent', 'pilot', 'zero_arm_current_gripper_hold')
    report = {'kind': 'CONTROL_GEOMETRY_POLICY_EVIDENCE_SYNTHESIS',
              'sources_sha256': {str(p): file_sha256(p) for p in
                                 (args.ik/'review.json', args.replay/'report.json', args.motion/'report.json')},
              'script_sha256': file_sha256(Path(__file__)), 'contact_timing': {}, 'reviewed_policy_phases': {}}
    for ep, case in replay['episodes'].items():
        rows = json.loads((args.replay / f'episode{ep}_follower_next_trace.json').read_text())
        contacts = json.loads((args.replay / f'episode{ep}_follower_next_contacts.json').read_text())
        images = json.loads((args.replay / f'episode{ep}_image_trace.json').read_text())
        origin = np.array(case['ball_xy'])
        real_origin = np.array(images[0]['real_ball_centroid_480px'][:2])
        sim_distance = [(r['time_s'], np.linalg.norm(np.array(r['ball_xyz_task'])[:2]-origin)) for r in rows]
        image_distance = [(r['time_s'], np.linalg.norm(np.array(r['real_ball_centroid_480px'][:2])-real_origin)
                          if r['real_ball_centroid_480px'] else -1) for r in images]
        first = contacts[0] if contacts else None
        if first:
            tick = max(1, int(np.ceil(first['time_s']*10)))
            row = rows[tick-1]
            report['contact_timing'][ep] = {
                'first_contact_s': first['time_s'], 'first_contact_geoms': first['pairs'],
                'pose_tracking_at_contact_mm': row['pose_error_m'] * 1000,
                'jaw_gap_at_contact_mm': row['jaw_tip_gap_m'] * 1000,
                'sim_ball_10mm_displacement_s': first_persistent(sim_distance, .01),
                'real_ball_8px_centroid_displacement_s': first_persistent(image_distance, 8.),
                'caveat': 'Three consecutive 10Hz samples; centroid displacement is a 2-D color/occlusion proxy, not a physical 3-D motion or contact label. Thresholds are descriptive, never controller thresholds.'}
            with av.open(str(args.replay/f'episode{ep}_complete.mp4')) as container:
                for k, frame in enumerate(container.decode(video=0)):
                    if k == tick:
                        cv2.imwrite(str(args.out/f'episode{ep}_first_contact.jpg'), frame.to_ndarray(format='bgr24'))
                        break
        windows = replay['review'][ep]
        for phase, (start, stop) in {'approach': [0, windows['closure'][0]], **windows}.items():
            mask = (preds['episode'] == int(ep)) & (preds['timestamp'] >= start) & (preds['timestamp'] < stop)
            report['reviewed_policy_phases'][f'ep{ep}_{phase}'] = {
                name: scores(preds[name][mask], preds['label'][mask], std) for name in names}
    report['decisions'] = [
        'IK deadband confirmed and fixed behind explicit numerical convergence settings; legacy configs preserve historical convergence defaults. Solver iteration budget enforced. No safety or acceptance threshold relaxed.',
        'Recorded relative deltas accumulate execution error. Privileged bounded reference feedback restores submillimetre closure/transport tracking in episodes 1 and 2 but remains imperfect in episode 3 and infeasible parts of the full trajectories.',
        'Follower joint replay tracks closure/transport well in episodes 1 and 2 but all three grasps have premature unilateral contact, zero two-jaw substeps and no held lift. Investigate camera/joint/gripper calibration, jaw collision geometry, real ball support height and contact compliance before attributing rollout failure to ACT.',
        'Real-to-sim pose alignment is not independently proven: reference FK, simulated tracking and label generation share the same provisional mapping. Real frames with projected jaw tips expose remaining visual discrepancies. Jaw tip gap is a model proxy, not a measured real opening.',
        'Frozen ACT beats zero motion on active held-out data. Pilot has slightly worse active-motion error than parent despite lower stationary error. This is no evidence of a general zero-motion collapse and no isolated proof of a policy/data coverage failure.',
        'No new training, labels, split changes, geometry/contact edits or production configuration adoption. Reference feedback is privileged diagnostics only; no ground-truth object or future-reference information enters ACT.']
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.out/'executed_script.txt').write_text(Path(__file__).read_text())
    # Relative links stay portable with the repository.
    def link(path):
        import os
        return os.path.relpath(path, args.out)
    lines = ['# IK, recorded-grasp and motion audit — 15 October 2026', '',
             'IK convergence is fixed in isolated opt-in runs. Grasp execution is still unresolved: accurate follower tracking does not reproduce the real grasps. Calibration/contact remains a confound, so ACT failure is not yet isolated to policy/data coverage.', '',
             'The final code passed 422 unit/integration tests covering IK, hard acceptance, collision/workspace rejection, joint limits, policy contracts and both robots. No checkpoint was trained or modified.', '',
             '## Frozen-policy control', '',
             'Three matched seeds per model/robot, 300 ticks each: 24 rollouts, 12 per convergence setting. Historical baseline trajectories reproduced within 1 micrometre; checkpoint hashes matched. Numerical convergence is 10 micrometres / 0.0001 rad. Position acceptance remains 3 mm, orientation acceptance 0.35 rad (SO-101) / 0.05 rad (Panda), penetration tolerance 2 mm; workspace, actuator, velocity and action limits are unchanged.', '',
             '| Policy / robot | Accepted unchanged arm, old → polished | Median requested / achieved movement, old → polished (mm/tick) | Median movement error, old → polished (mm) | Closest ball, old → polished (mm) |',
             '|---|---:|---:|---:|---:|']
    for name, variants in ik['cases'].items():
        old, new = variants['baseline'], variants['polished']
        movement = ' → '.join(f"{v['requested_translation_m']['median']*1000:.3f} / {v['achieved_translation_m']['median']*1000:.3f}" for v in (old,new))
        lines.append(f"| {name} | {old['accepted_unchanged_arm']} → {new['accepted_unchanged_arm']} / {new['ticks']} | {movement} | {old['tracking_error_m']['median']*1000:.3f} → {new['tracking_error_m']['median']*1000:.3f} | {old['mean_closest_ball_m']*1000:.1f} → {new['mean_closest_ball_m']*1000:.1f} |")
    lines += ['', 'All 12 polished trials still failed before a grasp. Policy inputs diverge once motion changes, so cross-run error changes are not a fixed-command comparison. Shadow IK solves on identical seeds/targets independently establish that formerly accepted no-op commands now move. Panda still achieves roughly half the requested per-tick displacement; tighter IK alone does not remove servo lag.', '',
              f"[Full frozen results and hashes]({link(args.ik/'review.json')}). Each run has raw, clamped and achieved six-dimensional deltas, solver residuals, rejections and joint-control traces.", '',
              '## Complete recorded grasps', '',
              'Episodes 1, 2 and 3 visually show a real grasp, carried ball, release and ball remaining in the bowl. Durations are 59.7, 34.4 and 45.6 seconds. Every recorded 10 Hz tick is replayed, including the portion beyond the old 30-second cutoff. These are diagnostic selections; the original unknown outcome annotations are unchanged.', '',
              'Five dynamic controls per episode: next follower state, recorded Cartesian deltas with legacy/polished IK, bounded feedback to the next recorded absolute pose, and polished deltas with the original leader-gripper labels. The first four use the next recorded follower gripper to isolate achieved jaw movement. The fifth retains the existing policy-label gripper semantics.', '',
              '| Episode | Follower closure / transport error (mm) | Cartesian old → polished closure / transport (mm) | Bounded reference feedback closure / transport (mm) |',
              '|---|---:|---:|---:|']
    for ep, data in replay['episodes'].items():
        def pair(mode, case=data):
            phases = case['modes'][mode]['phases']
            return ' / '.join(f"{phases[p]['pose_error_m']['mean']*1000:.2f}" for p in ('closure','lift_transport'))
        lines.append(f"| {ep} | {pair('follower_next')} | {pair('cartesian_legacy')} → {pair('cartesian_polished')} | {pair('cartesian_reference_feedback')} |")
    lines += ['', 'All 15 dynamic replays had zero two-jaw contact substeps and zero held lifts. Ball contact was inspected at every 500 Hz physics substep, so this result is not a 10 Hz sampling artifact. Reference feedback passes the existing IK gates on the good tracking segments; it does not bypass rejected poses. Whole-episode errors are larger: start/return references violate native joint limits and some recorded poses violate the collision/workspace gates.', '',
              '| Episode | First unilateral contact | Tracking error at contact (mm) | Sim ball moves >10 mm | Real centroid moves >8 px |',
              '|---|---|---:|---:|---:|']
    for ep, c in report['contact_timing'].items():
        names_c = ', '.join(p['geom'] for p in c['first_contact_geoms'])
        lines.append(f"| {ep} | {c['first_contact_s']:.3f}s, {names_c} | {c['pose_tracking_at_contact_mm']:.2f} | {c['sim_ball_10mm_displacement_s']:.1f}s | {c['real_ball_8px_centroid_displacement_s']:.1f}s |")
    lines += ['', 'The centroid criterion requires three consecutive frames and is a 2-D image-motion proxy, affected by jaw occlusion. It is not a measured real contact time. Nevertheless, the images and substep traces agree that the simulated ball is displaced before the recorded closure. The unchanged camera/joint/gripper mapping and contact model therefore need verification. The physical real jaw opening is not measured; reported gaps are the distance between the model’s two tip spheres.', '',
              'Using leader-gripper targets instead of follower openings creates roughly 48–50 mm mean jaw-gap error during transport in episodes 1 and 2. That is an execution/load/geometry diagnostic, not a reason to relabel the dataset automatically.', '',
              'The full videos show exact source frames, projected jaw tips, the recorded kinematic pose, follower dynamics and polished Cartesian dynamics. The reference pane hides the ball because its real 3-D path is unknown; it must not be read as a physical replay.', '']
    for ep in replay['episodes']:
        lines.append(f"- Episode {ep}: [complete video]({link(args.replay/f'episode{ep}_complete.mp4')}), [comparison sheet]({link(args.replay/f'episode{ep}_comparison.jpg')}), [first simulated contact](episode{ep}_first_contact.jpg).")
    lines += ['', f"[Full replay metrics, phase definitions and provenance]({link(args.replay/'report.json')}). Every mode also has joint states, per-tick tracking/jaw traces and substep contacts.", '',
              '## Active versus stationary predictions', '',
              '4,055 valid actions from nine held-out episodes are each scored once using the actual two-action execution prefix. Padding and terminal extrapolated labels are excluded. The zero baseline sends zero arm motion and holds the gripper observed at inference, without future-state information. Active means >1 mm translation or >0.01 rad rotation per 100 ms.', '',
              '| Stratum | Actions | Parent / pilot / zero translation error (mm) | Parent / pilot / zero normalized L1 |',
              '|---|---:|---:|---:|']
    for group in ('arm_active','arm_stationary','reviewed_stationary_ball_approach_proxy','observed_gripper_closure_proxy','upward_with_gripper_at_most_45pct_proxy'):
        g = motion['strata'][group]
        errors = ' / '.join(f"{g[n]['translation_vector_error_m']['mean']*1000:.2f}" for n in names)
        l1 = ' / '.join(f"{g[n]['mae_normalized']:.4f}" for n in names)
        lines.append(f"| {group} | {g['parent']['n']} | {errors} | {l1} |")
    lines += ['', 'Approach, closure and upward-motion rows above are kinematic proxies and can overlap. Additional errors for the reviewed approach/closure/transport/release windows of episodes 1–3 are in report.json, including rotation and gripper errors. Phase boundaries are coarse visual review, not ground-truth contact labels.', '',
              f"[Motion audit]({link(args.motion/'review.md')}) and [full predictions]({link(args.motion/'predictions.npz')}).", '',
              'Both ACT models beat zero motion on active real data. The pilot is slightly worse during active movement while improving stationary error. This does not establish policy/data coverage as the cause of the failed grasps. The next discriminating result is a physically successful recorded reference replay under a verified jaw/ball registration, followed by a frozen-policy comparison in that same setup.', '',
              '## Reproduce', '', '```sh',
              '.venv/bin/python scripts/diag_ik_polishing.py --out <new-ik-directory>',
              '.venv/bin/python scripts/diag_recorded_grasp_control.py --out <new-replay-directory>',
              '.venv/bin/python scripts/diag_motion_prediction_audit.py --out <new-motion-directory>',
              '.venv/bin/python scripts/summarize_control_audit.py --ik <ik-directory> --replay <replay-directory> --motion <motion-directory> --out <new-summary-directory>',
              '```', '',
              'Rendering requires macOS graphics access. Each run refuses to overwrite an existing output directory and records checksums. Executed script snapshots are stored as text alongside the reports. The verified directories contain the final iteration-budget fix; earlier audit directories retain the preliminary runs.', '']
    (args.out/'review.md').write_text('\n'.join(lines))
    print(args.out/'review.md')


if __name__ == '__main__':
    main()
