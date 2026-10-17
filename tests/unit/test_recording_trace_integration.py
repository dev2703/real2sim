"""Exercise staged recording telemetry without connecting to robot hardware."""

import __future__

import ast
import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).parents[2] / 'integrations/lerobot_trace/recording_trace.py'
SPEC = importlib.util.spec_from_file_location('recording_trace', SOURCE)
trace_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace_module)


def test_positions_are_copied_without_images_or_nonfinite_json():
    raw = {'shoulder.pos': np.float32(2), 'camera': np.zeros((2, 2, 3)), 'gripper.pos': float('nan')}
    result = trace_module.position_snapshot(raw)
    raw['shoulder.pos'] = 100
    assert result == {'shoulder.pos': 2., 'gripper.pos': None}
    json.dumps(result, allow_nan=False)


def test_tick_appends_do_not_write_to_disk(tmp_path, monkeypatch):
    trace = trace_module.RecordingTrace(tmp_path, {'dataset_fps': 20})
    trace.begin(0)
    def forbidden(*args):
        raise AssertionError('disk I/O inside append')
    monkeypatch.setattr(trace_module, '_write', forbidden)
    for i in range(50):
        trace.append({'tick_start_ns': i * 50_000_000})
    assert trace.active['frames'][-1]['frame_index'] == 49


def test_rerecord_keeps_discarded_attempt_and_saved_attempt_separate(tmp_path):
    trace = trace_module.RecordingTrace(tmp_path, {})
    for outcome in ['discarded', 'saved']:
        trace.begin(0)
        trace.append({'tick_start_ns': 100})
        trace.capture_finished()
        assert json.loads(trace.path.read_text())['status'] == 'captured_pending_dataset_save'
        trace.finish(outcome)
    trace.close()
    attempts = [json.loads(f.read_text()) for f in trace.root.glob('episode_*.json')]
    assert sorted(x['status'] for x in attempts) == ['discarded', 'saved']
    assert len({x['attempt_id'] for x in attempts}) == 2
    assert all(x['frames'][0]['frame_index'] == 0 for x in attempts)
    assert (trace.root/'COMPLETE.json').exists()
    assert not (trace.root/'RUNNING.json').exists()


def test_interrupted_trace_retains_partial_rows_and_flags_failure(tmp_path):
    trace = trace_module.RecordingTrace(tmp_path, {})
    trace.begin(3)
    trace.append({'sent_target': {'joint.pos': 2.}})
    path = trace.path
    trace.close()
    assert json.loads(path.read_text())['status'] == 'interrupted'
    assert (trace.root/'FAILED.json').exists()
    assert not (trace.root/'COMPLETE.json').exists()
    assert not (trace.root/'RUNNING.json').exists()


@pytest.mark.parametrize('enabled', [False, True])
def test_record_loop_preserves_commands_and_nominal_rows_while_tracing_actual_gaps(tmp_path, enabled):
    # Execute the installed collector, or the staged patch before installation.
    # A checkout without either local artifact can still run the helper unit tests.
    staged = Path(__file__).parents[2] / 'outputs/collection_trace_patch/lerobot_record.py'
    installed = Path(__file__).parents[3] / 'lerobot/src/lerobot/scripts/lerobot_record.py'
    source = installed if installed.is_file() and 'recording_trace:' in installed.read_text() else staged
    if not source.is_file():
        pytest.skip('Optional local collector integration requires an installed or staged patch')
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'record_loop')
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    clock = [0.]
    sent, stored = [], []

    class Leader:
        def get_action(self):
            return {'joint.pos': 10.}

    class Timer:
        def tick(self):
            pass

        def section(self, name):
            return contextlib.nullcontext()

        def wait(self):
            clock[0] += .1 if len(sent) == 1 else .05

    def observation():
        clock[0] += .001
        return {'joint.pos': 1., 'camera': np.zeros((2, 2, 3))}

    def send(action):
        sent.append(dict(action))
        clock[0] += .002
        return {'joint.pos': 2.}

    def mutate_processor(pair):
        pair[0]['joint.pos'] = 15.
        return pair[0]

    namespace = {'Teleoperator': Leader, 'time': SimpleNamespace(perf_counter=lambda: clock[0],
                  perf_counter_ns=lambda: int(clock[0]*1e9)), 'position_snapshot': trace_module.position_snapshot,
                 'OBS_STR': 'observation', 'ACTION': 'action',
                 'build_dataset_frame': lambda features, values, prefix: {prefix: dict(values)}}
    exec(compile(module, str(source), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    dataset = SimpleNamespace(fps=20, features={}, add_frame=stored.append)
    trace = trace_module.RecordingTrace(tmp_path, {}) if enabled else None
    if trace:
        trace.begin(0)
    namespace['record_loop'](robot=SimpleNamespace(name='so_follower', get_observation=observation,
        send_action=send), events={'exit_early': False}, fps=20,
        teleop_action_processor=mutate_processor, robot_action_processor=lambda pair: pair[0],
        robot_observation_processor=lambda obs: obs, dataset=dataset, teleop=Leader(),
        control_time_s=.15, timer=Timer(), recording_trace=trace)
    assert sent == [{'joint.pos': 15.}, {'joint.pos': 15.}]
    assert [row['action'] for row in stored] == [{'joint.pos': 2.}, {'joint.pos': 2.}]
    if trace:
        frames = json.loads(trace.path.read_text())['frames']
        assert len(frames) == len(stored)
        assert [x['frame_index'] for x in frames] == [0, 1]
        assert frames[1]['tick_start_ns'] - frames[0]['tick_start_ns'] > 100_000_000
        for frame in frames:
            assert frame['leader_request'] == {'joint.pos': 10.}
            assert frame['processed_request'] == {'joint.pos': 15.}
            assert frame['sent_target'] == {'joint.pos': 2.}
            assert frame['measured_positions'] == {'joint.pos': 1.}
            assert frame['observe_start_ns'] <= frame['observe_end_ns'] <= frame['send_start_ns']
            assert frame['send_start_ns'] <= frame['send_end_ns'] <= frame['record_end_ns']


def test_trace_lifecycle_rejects_overlapping_attempts(tmp_path):
    trace = trace_module.RecordingTrace(tmp_path, {})
    with pytest.raises(RuntimeError):
        trace.append({})
    trace.begin(0)
    with pytest.raises(RuntimeError):
        trace.begin(1)
    with pytest.raises(ValueError):
        trace.finish('success')
