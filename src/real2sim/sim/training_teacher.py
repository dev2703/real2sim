"""Simulator-state demonstrations for training; never a raw-policy controller."""

from __future__ import annotations

import numpy as np

from real2sim.policy.visual_waypoint import VisualWaypointController


class SimStateTrainingTeacher(VisualWaypointController):
    """Reuse native motion gates with simulator object poses instead of RGB detection.

    Privileged object/contact state is confined to demonstration generation.
    This teacher must never be constructed by the raw ACT evaluator.
    """

    controller_source = "SIM_STATE_TRAINING_TEACHER"

    def reset(self):
        super().reset()
        self._started = False

    def _acquire(self):
        env = self.env
        self.ball = env.data.xpos[env.ball_body].copy() - env.offset
        self.bowl = env.data.xpos[env.model.body("bowl").id].copy() - env.offset
        # Carry targets use the rim plane, as the original geometry planner did.
        self.bowl[2] += env.bowl["height"]
        if not np.isfinite(np.r_[self.ball, self.bowl]).all():
            raise ValueError("Training teacher received non-finite simulator object poses")
        self.perception_log.append({"tick": self.tick, "source": self.controller_source,
                                    "ball_task": self.ball.tolist(), "bowl_rim_task": self.bowl.tolist()})
        return True

    def step(self, obs, env):
        if not self._started:
            self._started = True
            self._acquire()
            # A corrective rollout may already hold the object. Restarting the
            # acquisition phase would reopen the jaw and destroy that state.
            metrics = env.info()["privileged_metrics"]
            if metrics["ball_held"]:
                self.grasp_rotation = env.ee_pose()[:3, :3].copy() if self.robot == "panda" else (
                    self._forward(self._native(), self._grip(False))[:3, :3].copy())
                self._transition("LOWER" if metrics["ball_over_bowl"] else (
                    "TRANSPORT" if metrics["lifted"] else "LIFT"))
        result = super().step(obs, env)
        result.detail["object_source"] = "privileged_simulator_state_for_training_only"
        result.detail["controller_source"] = self.controller_source
        for plan in self.plan_log:
            plan["uses_object_ground_truth"] = True
        return result
