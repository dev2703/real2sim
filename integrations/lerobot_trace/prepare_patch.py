"""Stage an exact-context patch against the user's current local recorder."""

import difflib
import hashlib
import json
from pathlib import Path

REPO = Path('/Users/deveshvyas/Downloads/lerobot')
OUT = Path('outputs/collection_trace_patch')
RECORDER = 'src/lerobot/scripts/lerobot_record.py'


def main():
    original = (REPO / RECORDER).read_text()
    changed = original

    def replace(old, new, count=1):
        nonlocal changed
        if changed.count(old) != count:
            raise ValueError(f'Unexpected recorder context count for {old[:80]!r}')
        changed = changed.replace(old, new)

    replace('from lerobot.utils.motor_temperature import MotorTemperatureMonitor, make_temperature_monitor\n',
            'from lerobot.utils.motor_temperature import MotorTemperatureMonitor, make_temperature_monitor\n'
            'from lerobot.utils.recording_trace import RecordingTrace, make_recording_trace, position_snapshot\n')
    replace('    resume: bool = False\n', '    resume: bool = False\n'
            '    # Optional local sidecars; host timing and command provenance, buffered outside control ticks.\n'
            '    recording_trace_dir: str | None = None\n')
    replace('    temperature_monitor: MotorTemperatureMonitor | None = None,\n',
            '    temperature_monitor: MotorTemperatureMonitor | None = None,\n'
            '    recording_trace: RecordingTrace | None = None,\n')
    replace('        timer.tick()\n\n        with timer.section("observe"):',
            '        timer.tick()\n'
            '        trace_row = {"tick_start_ns": time.perf_counter_ns()} if recording_trace else None\n\n'
            '        with timer.section("observe"):')
    replace('            obs = robot.get_observation()\n',
            '            if trace_row is not None:\n'
            '                trace_row["observe_start_ns"] = time.perf_counter_ns()\n'
            '            obs = robot.get_observation()\n'
            '            if trace_row is not None:\n'
            '                trace_row["observe_end_ns"] = time.perf_counter_ns()\n'
            '                trace_row["measured_positions"] = position_snapshot(obs)\n')
    replace('                act = teleop.get_action()\n',
            '                act = teleop.get_action()\n'
            '                if trace_row is not None:\n'
            '                    trace_row["leader_request"] = position_snapshot(act)\n')
    replace('                act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action\n',
            '                act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action\n'
            '                if trace_row is not None:\n'
            '                    trace_row["leader_request"] = position_snapshot(act)\n')
    replace('            _sent_action = robot.send_action(robot_action_to_send)\n',
            '            if trace_row is not None:\n'
            '                trace_row["processed_request"] = position_snapshot(robot_action_to_send)\n'
            '                trace_row["send_start_ns"] = time.perf_counter_ns()\n'
            '            _sent_action = robot.send_action(robot_action_to_send)\n'
            '            if trace_row is not None:\n'
            '                trace_row["send_end_ns"] = time.perf_counter_ns()\n'
            '                trace_row["sent_target"] = position_snapshot(_sent_action)\n')
    replace('                dataset.add_frame(frame)\n',
            '                dataset.add_frame(frame)\n'
            '            if trace_row is not None:\n'
            '                trace_row["record_end_ns"] = time.perf_counter_ns()\n'
            '                recording_trace.append(trace_row)\n')
    replace('\n\n@parser.wrap()\ndef record(',
            '\n    if recording_trace is not None:\n'
            '        recording_trace.capture_finished()\n\n\n@parser.wrap()\ndef record(')
    replace('    temperature_monitor = None\n', '    temperature_monitor = None\n    recording_trace = None\n')
    replace('        listener, events = init_keyboard_listener()\n',
            '        if cfg.recording_trace_dir is not None:\n'
            '            recording_trace = make_recording_trace(\n'
            '                cfg.recording_trace_dir, cfg, robot, teleop, __file__\n'
            '            )\n'
            '            logging.info("Local recording trace: %s", recording_trace.root)\n\n'
            '        listener, events = init_keyboard_listener()\n')
    replace('                log_say(f"Recording episode {episode_index}", cfg.play_sounds)\n',
            '                log_say(f"Recording episode {episode_index}", cfg.play_sounds)\n'
            '                if recording_trace is not None:\n'
            '                    recording_trace.begin(episode_index)\n')
    replace('                    timer=timer,\n                    temperature_monitor=temperature_monitor,\n',
            '                    timer=timer,\n                    temperature_monitor=temperature_monitor,\n'
            '                    recording_trace=recording_trace,\n')
    replace('                    dataset.clear_episode_buffer()\n',
            '                    dataset.clear_episode_buffer()\n'
            '                    if recording_trace is not None:\n'
            '                        recording_trace.finish("discarded")\n')
    replace('                dataset.save_episode()\n',
            '                dataset.save_episode()\n'
            '                if recording_trace is not None:\n'
            '                    recording_trace.finish("saved")\n')
    replace('    finally:\n        if temperature_monitor:',
            '    finally:\n        if recording_trace is not None:\n'
            '            try:\n                recording_trace.close()\n'
            '            except Exception:\n'
            '                logging.exception("Failed to save recording trace; continuing robot cleanup")\n'
            '        if temperature_monitor:')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'lerobot_record.original.py').write_text(original)
    (OUT/'lerobot_record.py').write_text(changed)
    sources = {RECORDER: (original, changed),
               'src/lerobot/utils/recording_trace.py': ('', Path('integrations/lerobot_trace/recording_trace.py').read_text())}
    patch = ''.join(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                               fromfile=f'a/{name}' if before else '/dev/null', tofile=f'b/{name}'))
                    for name,(before,after) in sources.items())
    (OUT/'recording_trace.patch').write_text(patch)
    (OUT/'manifest.json').write_text(json.dumps({'repository':str(REPO), 'original_recorder_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'patched_recorder_sha256':hashlib.sha256(changed.encode()).hexdigest(), 'scope':'Opt-in buffered telemetry only; existing sent-action patch retained; no hardware settings or commands changed.'},indent=2)+'\n')
    print(str(OUT/'recording_trace.patch'))


if __name__ == '__main__':
    main()
