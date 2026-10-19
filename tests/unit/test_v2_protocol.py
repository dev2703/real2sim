import copy
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from real2sim.config import load_yaml
from real2sim.data.conditions import stratified_split, validate_fixed_split
from real2sim.data.labels import ABSOLUTE_ACTION_NAMES, absolute_pose_actions
from real2sim.experiments.v2 import CAMERAS, experiment_matrix, final_summary, gate_failures
from real2sim.geometry.so3 import matrix_from_rot6d
from real2sim.policy.adapters import (
    AbsolutePoseActionAdapter,
    ApplyResult,
    ContractError,
    ObservationAdapter,
)
from real2sim.policy.sampling import ResumableBatches, restore_rng, rng_state
from real2sim.sim.cameras import add_link_cameras, camera_bindings
from tests.unit.test_oracle_schema import FakeNormalizer, _priv, _tracker, joint_contract


def annotations():
    return {'episodes':[{'episode_index':i, 'condition': 'baseline' if i<31 else 'ball_shift' if i<36 else 'bowl_shift',
                          'condition_status':'confirmed','shift_direction':'+x'} for i in range(41)]}


def test_stratification_requires_confirmed_conditions_and_directions():
    a=annotations()
    s=stratified_split(a,list(range(41)))
    assert [len(s[k]) for k in ('train','val','reserved')]==[29,6,6]
    assert s==stratified_split(a,list(range(41)))
    validate_fixed_split(s,list(range(41)))
    s['train'][0]=s['val'][0]
    with pytest.raises(ValueError,match='checksum'):
        validate_fixed_split(s,list(range(41)))
    a['episodes'][31]['shift_direction']=None
    with pytest.raises(ValueError,match='direction'):
        stratified_split(a,list(range(41)))
    a['episodes'][0]['condition_status']='guess'
    with pytest.raises(ValueError,match='confirmed'):
        stratified_split(a,list(range(41)))


def test_absolute_labels_next_pose_current_grip_and_terminal_mask():
    poses=np.repeat(np.eye(4)[None],3,axis=0)
    poses[:,0,3]=[0,.01,.04]
    action, mask=absolute_pose_actions(poses,np.array([10,30,70]))
    assert action.shape==(3,len(ABSOLUTE_ACTION_NAMES))
    np.testing.assert_allclose(action[:,0],[.01,.04,.04])
    np.testing.assert_allclose(action[:,-1],[.1,.3,.7])
    np.testing.assert_array_equal(mask,[False,False,True])
    for a in action:
        np.testing.assert_allclose(matrix_from_rot6d(a[3:9]),np.eye(3))
    with pytest.raises(ValueError):
        matrix_from_rot6d(np.zeros(6))


def test_ordered_two_camera_adapter_and_missing_view():
    c=joint_contract();c['camera_keys']=CAMERAS[::-1]
    ad=ObservationAdapter(c,FakeNormalizer(6))
    class Env:
        def real_state(self): return np.zeros(6)
    obs={k:np.full((3,192,256),i,dtype=np.float32) for i,k in enumerate(CAMERAS)}
    batch=ad.batch(obs,Env())
    assert [k for k in batch if k.startswith('observation.images.')]==CAMERAS[::-1]
    assert not torch.equal(batch[CAMERAS[0]],batch[CAMERAS[1]])
    obs.pop(CAMERAS[1])
    with pytest.raises(ContractError,match='lacks'):
        ad.batch(obs,Env())


def test_camera_must_be_rigidly_mounted_and_well_formed():
    task={'camera_iphone':{'key':CAMERAS[0]},'additional_cameras':[
        {'key':CAMERAS[1],'name':'wrist','fovy_deg':70,'mounts':{'so101':{
            'body':'wrist','position_body_m':[0,0,.01],'xyaxes_body':[1,0,0,0,1,0]}}}]}
    root=ET.fromstring('<mujoco><worldbody><body name="wrist"/></worldbody></mujoco>')
    add_link_cameras(root,task,'so101')
    assert root.find('.//body/camera').get('name')=='wrist'
    with pytest.raises(ValueError,match='missing panda mount'):
        camera_bindings(task,'panda')
    task['additional_cameras'][0]['mounts']['so101']['xyaxes_body']=[1,0,0,1,0,0]
    with pytest.raises(ValueError,match='orthonormal'):
        add_link_cameras(root,task,'so101')


def test_absolute_pose_uses_feedback_not_accumulated_deltas():
    class Delta:
        target_reference='current_pose'
        def __init__(self):
            self.env=self;self.position=.1
        def ee_pose(self):
            t=np.eye(4);t[0,3]=self.position;return t
        def apply(self,action):
            self.received=action.copy();return ApplyResult({}, {}, False, 0.)
    delta=Delta();adapter=AbsolutePoseActionAdapter(delta)
    action=np.array([.11,0,0,1,0,0,0,1,0,.5])
    adapter.apply(action)
    assert delta.received[0]==pytest.approx(.01)
    delta.position=.105
    adapter.apply(action)
    assert delta.received[0]==pytest.approx(.005)
    delta.target_reference='accepted_target'
    with pytest.raises(ValueError,match='current-pose'):
        AbsolutePoseActionAdapter(delta)


def test_resume_preserves_sample_order_and_stochastic_augmentation_across_epochs():
    class Dataset:
        def __len__(self): return 7
        def __getitem__(self,i): return torch.tensor([i,np.random.rand(),torch.rand(()).item()])
    first=ResumableBatches(Dataset(),3,42)
    next(first)
    state,randomness=first.state_dict(),rng_state()
    expected=[next(first) for _ in range(6)]
    resumed=ResumableBatches(Dataset(),3,999)
    resumed.load_state_dict(state);restore_rng(randomness)
    for result in expected:
        torch.testing.assert_close(next(resumed),result,rtol=0,atol=0)


def test_pushing_into_bowl_remains_historical_success_only():
    t=_tracker(success_hold_ticks=2)
    for _ in range(2):t.update(_priv(held=False,in_bowl=True,resting=True))
    assert t.success and not t.pick_place_success
    t=_tracker(success_hold_ticks=2)
    t.update(_priv(held=True,in_bowl=False,resting=False,lifted=True))
    for _ in range(2):t.update(_priv(held=False,in_bowl=True,resting=True))
    assert t.success and t.pick_place_success


def test_experiment_budgets_and_camera_controls():
    arms=experiment_matrix(load_yaml('configs/v2/act_base.yaml'))
    assert len([a for a in arms if a['stage']=='L1'])==3
    for arm in arms:
        t=arm['config']['train']
        assert t['steps']*t['batch_size']*t['grad_accum']==64000
        assert t['val_every']*t['batch_size']*t['grad_accum']==8000
        assert arm['config']['model']['chunk_size']==25
        assert t['exact_resume'] and not t['keep_only_best_and_last']
    assert len(gate_failures({}))==6


def test_final_aggregation_clusters_scenarios_across_training_seeds():
    rows=[dict(training_seed=s,scenario_id=i,condition=c,pick_place_success=i<18,timeout_s=45)
          for c in ('baseline','ball_shift','bowl_shift') for i in range(30) for s in range(3)]
    result=final_summary(rows)
    assert result['milestone_60_percent']
    assert result['mean_success']==pytest.approx(.6)
    assert result['n_scenarios']==90
    assert all(v==pytest.approx(.6) for v in result['per_seed'].values())
    bad=copy.deepcopy(rows);bad[-1]=bad[0]
    with pytest.raises(ValueError,match='duplicate'):
        final_summary(bad)
