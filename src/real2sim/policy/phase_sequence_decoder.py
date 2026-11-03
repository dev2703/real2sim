"""Causal phase filtering with a transition model fitted to training rollouts.

The transition graph and initial distribution come from recorded command-phase
labels. There is no hand-written phase order, simulator object state, or expert
phase input at evaluation. This is an optional learned sequence architecture;
it does not establish that a grasp or any other task event actually happened.
"""

from __future__ import annotations

import numpy as np

SCHEMA = "learned_phase_sequence_decoder_v1"


def fit_phase_sequence_decoder(collection_ids, robots, episodes, ticks, phases, *,
                               n_phases=9, transition_pseudocount=0., initial_pseudocount=0.):
    """Fit observed adjacent edges and actual tick-zero initial phases only.

    Rows from different collections, robots or episodes never share an edge.
    Missing ticks do not invent a transition. Optional pseudocounts apply only
    to observed support; absent transitions remain zero. A phase without any
    outgoing training edge receives an absorbing diagonal for normalization.
    """
    from real2sim.policy.phase_sequences import consecutive_pairs

    if (not isinstance(n_phases, (int, np.integer)) or n_phases < 1
            or not np.isfinite([transition_pseudocount, initial_pseudocount]).all()
            or min(transition_pseudocount, initial_pseudocount) < 0):
        raise ValueError("Positive phase count and finite nonnegative pseudocounts required")
    keys = [np.asarray(value) for value in (collection_ids, robots, episodes, ticks)]
    labels = np.asarray(phases)
    pairs = consecutive_pairs(*keys)
    if (labels.shape != keys[0].shape or not len(labels)
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0) or np.any(labels >= n_phases)):
        raise ValueError("Training phases must be matching nonempty integer IDs in range")
    initial = labels[keys[3] == 0]
    if not len(initial):
        raise ValueError("Sequence initialization requires recorded tick-zero training rows")
    counts = np.zeros((n_phases, n_phases), np.int64)
    if len(pairs):
        np.add.at(counts, (labels[pairs[:, 0]], labels[pairs[:, 1]]), 1)
    initial_counts = np.bincount(initial, minlength=n_phases)
    transition = counts.astype(np.float64)
    transition[counts > 0] += transition_pseudocount
    empty_rows = counts.sum(-1) == 0
    transition[np.flatnonzero(empty_rows), np.flatnonzero(empty_rows)] = 1.
    transition /= transition.sum(-1, keepdims=True)
    prior = initial_counts.astype(np.float64)
    prior[initial_counts > 0] += initial_pseudocount
    prior /= prior.sum()
    return {"schema": SCHEMA, "phase_count": int(n_phases),
            "transition_counts": counts.tolist(), "initial_counts": initial_counts.tolist(),
            "transition_matrix": transition.tolist(), "initial_prior": prior.tolist(),
            "transition_pseudocount": float(transition_pseudocount),
            "initial_pseudocount": float(initial_pseudocount),
            "pseudocount_support": "Observed edges and recorded tick-zero phases only",
            "unseen_outgoing_rows": np.flatnonzero(empty_rows).tolist(),
            "unseen_row_normalization": "Absorbing diagonal; no invented outgoing edges",
            "training_rows": len(labels), "consecutive_pairs": len(pairs),
            "tick_zero_rows": len(initial),
            "training_only": True, "teacher_phase_is_runtime_input": False}


class LearnedPhaseSequenceDecoder:
    """Filter one online neural phase distribution per consecutive control tick.

    The first update uses the learned initial prior. Later updates propagate
    the preceding posterior through the learned transition matrix before
    applying neural emissions. Reset at every episode boundary. The returned
    probabilities must also route the action components; changing only the
    reported phase would leave execution inconsistent with the decoder.
    """

    def __init__(self, metadata, *, emission_floor=1e-8, emission_power=1.):
        if metadata.get("schema") != SCHEMA:
            raise ValueError("Unsupported learned phase sequence artifact")
        transition = np.asarray(metadata.get("transition_matrix"), dtype=np.float64)
        prior = np.asarray(metadata.get("initial_prior"), dtype=np.float64)
        n_phases = metadata.get("phase_count")
        if (not isinstance(n_phases, int) or n_phases < 1
                or transition.shape != (n_phases, n_phases) or prior.shape != (n_phases,)
                or not np.isfinite(transition).all() or not np.isfinite(prior).all()
                or np.any(transition < 0) or np.any(prior < 0)
                or not np.allclose(transition.sum(-1), 1., rtol=0., atol=1e-10)
                or not np.isclose(prior.sum(), 1., rtol=0., atol=1e-10)):
            raise ValueError("Sequence decoder requires finite stochastic transition/prior arrays")
        if (not np.isfinite([emission_floor, emission_power]).all()
                or not 0 < emission_floor < 1 or emission_power <= 0):
            raise ValueError("Sequence emissions require a floor in (0,1) and positive power")
        self.transition = transition.copy()
        self.initial_prior = prior.copy()
        self.n_phases = n_phases
        self.emission_floor, self.emission_power = float(emission_floor), float(emission_power)
        self.reset()

    def reset(self):
        self._posterior = None
        self.steps = 0
        self.last_phase_id = None
        self.last_diagnostics = None

    @property
    def posterior(self):
        return (self.initial_prior if self._posterior is None else self._posterior).copy()

    def step(self, emissions):
        emissions = np.asarray(emissions, dtype=np.float64)
        if (emissions.shape != (self.n_phases,) or not np.isfinite(emissions).all()
                or np.any(emissions < 0) or not np.any(emissions > 0)):
            raise ValueError("Sequence decoder needs one finite nonnegative neural phase distribution")
        emissions = emissions / emissions.max()
        emissions /= emissions.sum()
        prior = (self.initial_prior.copy() if self._posterior is None
                 else self._posterior @ self.transition)
        # Work in log space to avoid underflow with sharp neural emissions.
        # Zero learned support stays exactly zero; the emission floor cannot
        # create a transition that was absent from the training graph.
        supported = prior > 0
        log_posterior = np.full(self.n_phases, -np.inf)
        log_emissions = np.log(np.maximum(emissions[supported], self.emission_floor))
        log_emissions -= log_emissions.max()
        with np.errstate(over="ignore"):
            log_posterior[supported] = np.log(prior[supported]) + self.emission_power * log_emissions
        posterior = np.exp(log_posterior - log_posterior.max())
        posterior /= posterior.sum()
        self._posterior = posterior
        self.last_phase_id = int(posterior.argmax())
        self.last_diagnostics = {"step": self.steps, "predicted_prior": prior.tolist(),
            "neural_emissions": emissions.tolist(), "filtered_posterior": posterior.tolist(),
            "selected_phase_id": self.last_phase_id,
            "teacher_phase_is_runtime_input": False}
        self.steps += 1
        return posterior.copy()
