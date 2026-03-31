from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np

from .datasets import STIMULUS_NAMES, conditioning_query, generate_second_order_conditioning


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return -np.inf
    max_value = np.max(values)
    if not np.isfinite(max_value):
        return float(max_value)
    return float(max_value + np.log(np.sum(np.exp(values - max_value))))


def bernoulli_log_prob_from_logits(y: np.ndarray, logits: np.ndarray) -> np.ndarray:
    return y * logits - np.logaddexp(0.0, logits)


def normal_logpdf(x: np.ndarray | float, mean: float, std: float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    var = std * std
    log_norm = -0.5 * np.log(2.0 * np.pi * var)
    value = log_norm - 0.5 * ((x_arr - mean) ** 2) / var
    if np.isscalar(x):
        return float(value)
    return value


@dataclass
class CourvillePrior:
    weight_std: float = 3.0
    latent_bias_std: float = 3.0
    stimulus_bias_mean: float = -15.0
    stimulus_bias_std: float = 1.0
    structure_penalty: float = 3.0
    max_causes: int = 5
    max_edges_per_cause: int | None = None

    def log_num_causes(self, num_causes: int) -> float:
        if num_causes < 0 or num_causes > self.max_causes:
            return -np.inf
        support = np.arange(self.max_causes + 1, dtype=float)
        log_unnormalized = -self.structure_penalty * np.log(10.0) * support
        return float(log_unnormalized[num_causes] - logsumexp(log_unnormalized))

    def max_edges(self, num_stimuli: int) -> int:
        if self.max_edges_per_cause is None:
            return num_stimuli
        return min(self.max_edges_per_cause, num_stimuli)

    def log_num_edges(self, num_edges: int, num_stimuli: int) -> float:
        max_edges = self.max_edges(num_stimuli)
        if num_edges < 0 or num_edges > max_edges:
            return -np.inf
        support = np.arange(max_edges + 1, dtype=float)
        log_unnormalized = -self.structure_penalty * np.log(10.0) * support
        return float(log_unnormalized[num_edges] - logsumexp(log_unnormalized))

    def sample_num_edges(self, rng: np.random.Generator, num_stimuli: int) -> int:
        max_edges = self.max_edges(num_stimuli)
        support = np.arange(max_edges + 1, dtype=int)
        log_probs = np.array([self.log_num_edges(int(k), num_stimuli) for k in support], dtype=float)
        probs = np.exp(log_probs - logsumexp(log_probs))
        return int(rng.choice(support, p=probs))


@dataclass
class SBNState:
    edge_mask: np.ndarray
    weights: np.ndarray
    latent_bias: np.ndarray
    stimulus_bias: np.ndarray

    @property
    def num_causes(self) -> int:
        return int(self.latent_bias.shape[0])

    def copy(self) -> "SBNState":
        return SBNState(
            edge_mask=self.edge_mask.copy(),
            weights=self.weights.copy(),
            latent_bias=self.latent_bias.copy(),
            stimulus_bias=self.stimulus_bias.copy(),
        )

    def effective_weights(self) -> np.ndarray:
        return self.weights * self.edge_mask


@dataclass
class PosteriorSample:
    num_causes: int
    edge_mask: np.ndarray
    weights: np.ndarray
    latent_bias: np.ndarray
    stimulus_bias: np.ndarray
    latent_states: np.ndarray
    log_posterior: float


class RJMCMCSigmoidBeliefNetwork:
    def __init__(
        self,
        observations: np.ndarray,
        prior: CourvillePrior | None = None,
        init_causes: int = 1,
        init_strategy: str = "prior",
        inverse_temperature: float = 1.0,
        parameter_step_scales: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.observations = np.asarray(observations, dtype=np.int8)
        if self.observations.ndim != 2:
            raise ValueError("observations must be a 2D binary array")
        if not np.all((self.observations == 0) | (self.observations == 1)):
            raise ValueError("observations must contain only 0/1 values")

        self.num_trials, self.num_stimuli = self.observations.shape
        self.prior = prior or CourvillePrior()
        self.rng = np.random.default_rng(seed)
        self.inverse_temperature = float(inverse_temperature)
        self.parameter_step_scales = {
            "weight": 0.75,
            "latent_bias": 0.6,
            "stimulus_bias": 0.6,
            "selectivity": 0.6,
        }
        if parameter_step_scales is not None:
            self.parameter_step_scales.update(parameter_step_scales)

        init_causes = int(np.clip(init_causes, 0, self.prior.max_causes))
        self.state = self.initialize_state(init_causes, init_strategy=init_strategy)
        self.current_log_prior = self.log_prior(self.state)
        self.current_log_likelihood = self.log_likelihood(self.state)
        self.current_log_posterior = self.current_log_prior + self.inverse_temperature * self.current_log_likelihood

        self.samples: list[PosteriorSample] = []
        self.trace: list[dict[str, float | int | str]] = []

    def initialize_state(self, num_causes: int, init_strategy: str) -> SBNState:
        if init_strategy == "prior":
            return self.sample_prior_state(num_causes)
        if init_strategy == "empirical_patterns":
            return self.sample_empirical_pattern_state(num_causes)
        raise ValueError(f"unknown init_strategy: {init_strategy}")

    def sample_prior_state(self, num_causes: int) -> SBNState:
        edge_mask = np.zeros((num_causes, self.num_stimuli), dtype=bool)
        weights = np.zeros((num_causes, self.num_stimuli), dtype=float)

        for cause_idx in range(num_causes):
            num_edges = self.prior.sample_num_edges(self.rng, self.num_stimuli)
            if num_edges > 0:
                edge_indices = self.rng.choice(self.num_stimuli, size=num_edges, replace=False)
                edge_mask[cause_idx, edge_indices] = True
                weights[cause_idx, edge_indices] = self.rng.normal(
                    0.0,
                    self.prior.weight_std,
                    size=num_edges,
                )

        latent_bias = self.rng.normal(0.0, self.prior.latent_bias_std, size=num_causes)
        stimulus_bias = self.rng.normal(
            self.prior.stimulus_bias_mean,
            self.prior.stimulus_bias_std,
            size=self.num_stimuli,
        )
        return SBNState(
            edge_mask=edge_mask,
            weights=weights,
            latent_bias=latent_bias,
            stimulus_bias=stimulus_bias,
        )

    def sample_empirical_pattern_state(self, num_causes: int) -> SBNState:
        if num_causes == 0:
            return SBNState(
                edge_mask=np.zeros((0, self.num_stimuli), dtype=bool),
                weights=np.zeros((0, self.num_stimuli), dtype=float),
                latent_bias=np.zeros(0, dtype=float),
                stimulus_bias=self.empirical_stimulus_bias(),
            )

        unique_rows, counts = np.unique(self.observations, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        chosen_rows = unique_rows[order[:num_causes]]
        chosen_counts = counts[order[:num_causes]]

        edge_mask = np.ones((len(chosen_rows), self.num_stimuli), dtype=bool)
        weights = np.zeros((len(chosen_rows), self.num_stimuli), dtype=float)
        for idx, row in enumerate(chosen_rows):
            cue_weights = np.where(row[:-1] == 1, 8.0, -8.0)
            us_weight = 12.0 if row[-1] == 1 else -12.0
            weights[idx] = np.concatenate([cue_weights, [us_weight]])

        latent_bias = np.log(np.clip(chosen_counts / self.num_trials, 1e-3, 1 - 1e-3))
        stimulus_bias = self.empirical_stimulus_bias()
        return SBNState(
            edge_mask=edge_mask,
            weights=weights,
            latent_bias=latent_bias,
            stimulus_bias=stimulus_bias,
        )

    def empirical_stimulus_bias(self) -> np.ndarray:
        rates = np.clip(self.observations.mean(axis=0), 1e-3, 1 - 1e-3)
        empirical = np.log(rates / (1.0 - rates))
        return np.minimum(empirical, np.full(self.num_stimuli, -4.0))

    def latent_configurations(self, num_causes: int) -> np.ndarray:
        if num_causes == 0:
            return np.zeros((1, 0), dtype=np.int8)
        indices = np.arange(2**num_causes, dtype=np.uint32)[:, None]
        bit_positions = np.arange(num_causes, dtype=np.uint32)[None, :]
        return ((indices >> bit_positions) & 1).astype(np.int8)

    def log_prior(self, state: SBNState) -> float:
        logp = self.prior.log_num_causes(state.num_causes)
        if not np.isfinite(logp):
            return -np.inf

        max_edges = self.prior.max_edges(self.num_stimuli)
        edge_counts = state.edge_mask.sum(axis=1) if state.num_causes > 0 else np.zeros(0, dtype=int)

        if np.any(edge_counts > max_edges):
            return -np.inf

        for edge_count in edge_counts:
            logp += self.prior.log_num_edges(int(edge_count), self.num_stimuli)
            logp -= math.log(math.comb(self.num_stimuli, int(edge_count)))

        logp += float(np.sum(normal_logpdf(state.stimulus_bias, self.prior.stimulus_bias_mean, self.prior.stimulus_bias_std)))
        if state.num_causes > 0:
            logp += float(np.sum(normal_logpdf(state.latent_bias, 0.0, self.prior.latent_bias_std)))
            active_weights = state.weights[state.edge_mask]
            if active_weights.size > 0:
                logp += float(np.sum(normal_logpdf(active_weights, 0.0, self.prior.weight_std)))
            if not np.allclose(state.weights[~state.edge_mask], 0.0):
                return -np.inf
        return float(logp)

    def trial_log_marginal(self, observation: np.ndarray, state: SBNState) -> float:
        x = self.latent_configurations(state.num_causes)
        latent_log_prob = np.sum(
            bernoulli_log_prob_from_logits(x, np.broadcast_to(state.latent_bias, x.shape)),
            axis=1,
        )
        obs_logits = x @ state.effective_weights() + state.stimulus_bias
        obs_log_prob = np.sum(bernoulli_log_prob_from_logits(observation[None, :], obs_logits), axis=1)
        return logsumexp(latent_log_prob + obs_log_prob)

    def log_likelihood(self, state: SBNState) -> float:
        total = 0.0
        for trial in self.observations:
            total += self.trial_log_marginal(trial, state)
        return float(total)

    def log_posterior(self, state: SBNState) -> float:
        return self.log_target(state)

    def log_target(self, state: SBNState, inverse_temperature: float | None = None) -> float:
        log_prior = self.log_prior(state)
        if not np.isfinite(log_prior):
            return -np.inf
        beta = self.inverse_temperature if inverse_temperature is None else float(inverse_temperature)
        return float(log_prior + beta * self.log_likelihood(state))

    def birth_proposal_logpdf(
        self,
        edge_mask_row: np.ndarray,
        weight_row: np.ndarray,
        latent_bias: float,
    ) -> float:
        edge_count = int(edge_mask_row.sum())
        logq = self.prior.log_num_edges(edge_count, self.num_stimuli)
        logq -= math.log(math.comb(self.num_stimuli, edge_count))
        logq += normal_logpdf(latent_bias, 0.0, self.prior.latent_bias_std)
        if edge_count > 0:
            logq += float(np.sum(normal_logpdf(weight_row[edge_mask_row], 0.0, self.prior.weight_std)))
        return float(logq)

    def parameter_move(self) -> tuple[SBNState, str, float]:
        families = ["stimulus_bias"]
        if self.state.num_causes > 0:
            families.append("latent_bias")
            if self.state.edge_mask.any():
                families.append("selectivity")
        if self.state.edge_mask.any():
            families.append("weight")

        family = str(self.rng.choice(families))
        proposal = self.state.copy()

        if family == "stimulus_bias":
            idx = int(self.rng.integers(self.num_stimuli))
            proposal.stimulus_bias[idx] += self.rng.normal(0.0, self.parameter_step_scales["stimulus_bias"])
        elif family == "latent_bias":
            idx = int(self.rng.integers(self.state.num_causes))
            proposal.latent_bias[idx] += self.rng.normal(0.0, self.parameter_step_scales["latent_bias"])
        elif family == "selectivity":
            cause_idx = int(self.rng.integers(self.state.num_causes))
            active = proposal.edge_mask[cause_idx]
            if np.any(active):
                delta = self.rng.normal(0.0, self.parameter_step_scales["selectivity"])
                proposal.latent_bias[cause_idx] -= delta
                signs = np.sign(proposal.weights[cause_idx, active])
                signs[signs == 0.0] = 1.0
                proposal.weights[cause_idx, active] += delta * signs
        else:
            active_edges = np.argwhere(self.state.edge_mask)
            edge_idx = tuple(active_edges[int(self.rng.integers(len(active_edges)))])
            proposal.weights[edge_idx] += self.rng.normal(0.0, self.parameter_step_scales["weight"])

        return proposal, f"parameter:{family}", 0.0

    def edge_move(self) -> tuple[SBNState, str, float] | None:
        if self.state.num_causes == 0:
            return None

        edge_counts = self.state.edge_mask.sum(axis=1)
        max_edges = self.prior.max_edges(self.num_stimuli)
        addable_positions = np.argwhere(~self.state.edge_mask & (edge_counts[:, None] < max_edges))
        active_positions = np.argwhere(self.state.edge_mask)

        can_add = len(addable_positions) > 0
        can_delete = len(active_positions) > 0
        if not can_add and not can_delete:
            return None

        if can_add and can_delete:
            move = "edge_add" if self.rng.random() < 0.5 else "edge_delete"
        else:
            move = "edge_add" if can_add else "edge_delete"

        if move == "edge_add":
            selected = tuple(addable_positions[int(self.rng.integers(len(addable_positions)))])
            proposal = self.state.copy()
            proposal.edge_mask[selected] = True
            proposal.weights[selected] = self.rng.normal(0.0, self.prior.weight_std)

            reverse_active = int(proposal.edge_mask.sum())
            forward_logq = -math.log(len(addable_positions)) + normal_logpdf(proposal.weights[selected], 0.0, self.prior.weight_std)
            reverse_logq = -math.log(reverse_active)
            return proposal, move, float(reverse_logq - forward_logq)

        selected = tuple(active_positions[int(self.rng.integers(len(active_positions)))])
        removed_weight = float(self.state.weights[selected])
        proposal = self.state.copy()
        proposal.edge_mask[selected] = False
        proposal.weights[selected] = 0.0

        reverse_counts = proposal.edge_mask.sum(axis=1)
        reverse_addable = np.argwhere(~proposal.edge_mask & (reverse_counts[:, None] < max_edges))
        forward_logq = -math.log(len(active_positions))
        reverse_logq = -math.log(len(reverse_addable)) + normal_logpdf(removed_weight, 0.0, self.prior.weight_std)
        return proposal, move, float(reverse_logq - forward_logq)

    def add_cause_move(self) -> tuple[SBNState, str, float] | None:
        if self.state.num_causes >= self.prior.max_causes:
            return None

        num_edges = self.prior.sample_num_edges(self.rng, self.num_stimuli)
        new_edge_mask = np.zeros(self.num_stimuli, dtype=bool)
        new_weight_row = np.zeros(self.num_stimuli, dtype=float)
        if num_edges > 0:
            chosen = self.rng.choice(self.num_stimuli, size=num_edges, replace=False)
            new_edge_mask[chosen] = True
            new_weight_row[chosen] = self.rng.normal(0.0, self.prior.weight_std, size=num_edges)
        new_latent_bias = float(self.rng.normal(0.0, self.prior.latent_bias_std))

        proposal = SBNState(
            edge_mask=np.vstack([self.state.edge_mask, new_edge_mask]),
            weights=np.vstack([self.state.weights, new_weight_row]),
            latent_bias=np.concatenate([self.state.latent_bias, [new_latent_bias]]),
            stimulus_bias=self.state.stimulus_bias.copy(),
        )

        forward_logq = self.birth_proposal_logpdf(new_edge_mask, new_weight_row, new_latent_bias)
        reverse_logq = -math.log(proposal.num_causes)
        return proposal, "cause_add", float(reverse_logq - forward_logq)

    def delete_cause_move(self) -> tuple[SBNState, str, float] | None:
        if self.state.num_causes == 0:
            return None

        delete_idx = int(self.rng.integers(self.state.num_causes))
        removed_edge_mask = self.state.edge_mask[delete_idx].copy()
        removed_weight_row = self.state.weights[delete_idx].copy()
        removed_latent_bias = float(self.state.latent_bias[delete_idx])

        proposal = SBNState(
            edge_mask=np.delete(self.state.edge_mask, delete_idx, axis=0),
            weights=np.delete(self.state.weights, delete_idx, axis=0),
            latent_bias=np.delete(self.state.latent_bias, delete_idx),
            stimulus_bias=self.state.stimulus_bias.copy(),
        )

        forward_logq = -math.log(self.state.num_causes)
        reverse_logq = self.birth_proposal_logpdf(removed_edge_mask, removed_weight_row, removed_latent_bias)
        return proposal, "cause_delete", float(reverse_logq - forward_logq)

    def sample_latent_states(self, state: SBNState) -> np.ndarray:
        x = self.latent_configurations(state.num_causes)
        if state.num_causes == 0:
            return np.zeros((self.num_trials, 0), dtype=np.int8)

        latent_log_prob = np.sum(
            bernoulli_log_prob_from_logits(x, np.broadcast_to(state.latent_bias, x.shape)),
            axis=1,
        )
        obs_logits = x @ state.effective_weights() + state.stimulus_bias

        samples = np.zeros((self.num_trials, state.num_causes), dtype=np.int8)
        for trial_idx, observation in enumerate(self.observations):
            obs_log_prob = np.sum(bernoulli_log_prob_from_logits(observation[None, :], obs_logits), axis=1)
            log_weights = latent_log_prob + obs_log_prob
            log_norm = logsumexp(log_weights)
            probs = np.exp(log_weights - log_norm)
            state_idx = int(self.rng.choice(len(x), p=probs))
            samples[trial_idx] = x[state_idx]
        return samples

    def conditional_stimulus_probabilities(
        self,
        state: SBNState,
        observed_values: np.ndarray,
        observed_mask: np.ndarray,
    ) -> np.ndarray:
        observed_values = np.asarray(observed_values, dtype=np.int8)
        observed_mask = np.asarray(observed_mask, dtype=bool)
        if observed_values.shape != (self.num_stimuli,) or observed_mask.shape != (self.num_stimuli,):
            raise ValueError("observed_values and observed_mask must have shape (num_stimuli,)")

        x = self.latent_configurations(state.num_causes)
        latent_log_prob = np.sum(
            bernoulli_log_prob_from_logits(x, np.broadcast_to(state.latent_bias, x.shape)),
            axis=1,
        )
        obs_logits = x @ state.effective_weights() + state.stimulus_bias
        if np.any(observed_mask):
            masked_logits = obs_logits[:, observed_mask]
            masked_values = observed_values[observed_mask][None, :]
            latent_log_prob = latent_log_prob + np.sum(
                bernoulli_log_prob_from_logits(masked_values, masked_logits),
                axis=1,
            )

        log_norm = logsumexp(latent_log_prob)
        posterior_x = np.exp(latent_log_prob - log_norm)
        probabilities = posterior_x @ sigmoid(obs_logits)
        probabilities = np.asarray(probabilities, dtype=float)
        probabilities[observed_mask] = observed_values[observed_mask]
        return probabilities

    def posterior_predictive(
        self,
        observed_values: np.ndarray,
        observed_mask: np.ndarray,
        samples: list[PosteriorSample] | None = None,
    ) -> np.ndarray:
        use_samples = self.samples if samples is None else samples
        if not use_samples:
            raise ValueError("no stored posterior samples are available")

        probs = []
        for sample in use_samples:
            state = SBNState(
                edge_mask=sample.edge_mask,
                weights=sample.weights,
                latent_bias=sample.latent_bias,
                stimulus_bias=sample.stimulus_bias,
            )
            probs.append(self.conditional_stimulus_probabilities(state, observed_values, observed_mask))
        return np.mean(np.asarray(probs), axis=0)

    def state_summary(self) -> dict[str, object]:
        return {
            "num_causes": self.state.num_causes,
            "edge_mask": self.state.edge_mask.astype(int).tolist(),
            "weights": np.round(self.state.effective_weights(), 2).tolist(),
            "latent_bias": np.round(self.state.latent_bias, 2).tolist(),
            "stimulus_bias": np.round(self.state.stimulus_bias, 2).tolist(),
        }

    def step(self) -> dict[str, float | int | str]:
        move_selector = int(self.rng.integers(4))
        if move_selector == 0:
            proposal_tuple = self.parameter_move()
        elif move_selector == 1:
            proposal_tuple = self.edge_move()
        elif move_selector == 2:
            proposal_tuple = self.add_cause_move()
        else:
            proposal_tuple = self.delete_cause_move()

        if proposal_tuple is None:
            record = {
                "move": "unavailable",
                "accepted": 0,
                "num_causes": self.state.num_causes,
                "log_posterior": self.current_log_posterior,
            }
            self.trace.append(record)
            return record

        proposal, move_name, log_proposal_ratio = proposal_tuple
        proposal_log_prior = self.log_prior(proposal)
        if np.isfinite(proposal_log_prior):
            proposal_log_likelihood = self.log_likelihood(proposal)
            proposal_log_posterior = proposal_log_prior + self.inverse_temperature * proposal_log_likelihood
        else:
            proposal_log_likelihood = -np.inf
            proposal_log_posterior = -np.inf
        log_acceptance = proposal_log_posterior - self.current_log_posterior + log_proposal_ratio

        accepted = int(np.log(self.rng.random()) < min(0.0, log_acceptance))
        if accepted:
            self.state = proposal
            self.current_log_prior = proposal_log_prior
            self.current_log_likelihood = proposal_log_likelihood
            self.current_log_posterior = proposal_log_posterior

        record = {
            "move": move_name,
            "accepted": accepted,
            "num_causes": self.state.num_causes,
            "log_posterior": self.current_log_posterior,
        }
        self.trace.append(record)
        return record

    def run(
        self,
        iterations: int,
        burn_in: int = 0,
        thin: int = 1,
        store_latents: bool = True,
        verbose_every: int = 0,
    ) -> list[PosteriorSample]:
        self.samples = []
        self.trace = []
        for iteration in range(1, iterations + 1):
            record = self.step()
            should_store = iteration > burn_in and ((iteration - burn_in) % thin == 0)
            if should_store:
                latent_states = self.sample_latent_states(self.state) if store_latents else np.zeros(
                    (self.num_trials, self.state.num_causes),
                    dtype=np.int8,
                )
                self.samples.append(
                    PosteriorSample(
                        num_causes=self.state.num_causes,
                        edge_mask=self.state.edge_mask.copy(),
                        weights=self.state.weights.copy(),
                        latent_bias=self.state.latent_bias.copy(),
                        stimulus_bias=self.state.stimulus_bias.copy(),
                        latent_states=latent_states,
                        log_posterior=float(record["log_posterior"]),
                    )
                )
            if verbose_every and iteration % verbose_every == 0:
                accept_rate = float(np.mean([entry["accepted"] for entry in self.trace]))
                print(
                    f"iter={iteration} move={record['move']} accepted={record['accepted']} "
                    f"C={self.state.num_causes} logpost={self.current_log_posterior:.2f} "
                    f"accept_rate={accept_rate:.3f}"
                )
        return self.samples


class ParallelTemperingRJMCMC:
    def __init__(
        self,
        observations: np.ndarray,
        inverse_temperatures: tuple[float, ...] = (1.0, 0.6, 0.35, 0.2),
        prior: CourvillePrior | None = None,
        init_causes: int = 2,
        init_strategy: str = "empirical_patterns",
        parameter_step_scales: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.observations = np.asarray(observations, dtype=np.int8)
        self.inverse_temperatures = tuple(float(beta) for beta in inverse_temperatures)
        self.rng = np.random.default_rng(seed)
        self.samplers = [
            RJMCMCSigmoidBeliefNetwork(
                self.observations,
                prior=prior,
                init_causes=init_causes,
                init_strategy=init_strategy,
                inverse_temperature=beta,
                parameter_step_scales=parameter_step_scales,
                seed=None if seed is None else seed + 1000 + chain_idx,
            )
            for chain_idx, beta in enumerate(self.inverse_temperatures)
        ]
        self.swap_attempts = 0
        self.swap_accepts = 0
        self.trace: list[dict[str, object]] = []
        self.samples: list[PosteriorSample] = []

    @property
    def cold_sampler(self) -> RJMCMCSigmoidBeliefNetwork:
        return self.samplers[0]

    def attempt_swaps(self) -> int:
        accepted = 0
        parity = int(self.rng.integers(2))
        for idx in range(parity, len(self.samplers) - 1, 2):
            left = self.samplers[idx]
            right = self.samplers[idx + 1]
            log_alpha = (
                (left.inverse_temperature - right.inverse_temperature)
                * (right.current_log_likelihood - left.current_log_likelihood)
            )
            self.swap_attempts += 1
            if np.log(self.rng.random()) < min(0.0, log_alpha):
                (
                    left.state,
                    right.state,
                    left.current_log_prior,
                    right.current_log_prior,
                    left.current_log_likelihood,
                    right.current_log_likelihood,
                ) = (
                    right.state,
                    left.state,
                    right.current_log_prior,
                    left.current_log_prior,
                    right.current_log_likelihood,
                    left.current_log_likelihood,
                )
                left.current_log_posterior = left.current_log_prior + left.inverse_temperature * left.current_log_likelihood
                right.current_log_posterior = right.current_log_prior + right.inverse_temperature * right.current_log_likelihood
                accepted += 1
                self.swap_accepts += 1
        return accepted

    def verbose_summary(self, iteration: int) -> str:
        cold = self.cold_sampler
        query_strings = []
        for label, present in (
            ("A", ("A",)),
            ("X", ("X",)),
            ("B", ("B",)),
            ("XB", ("X", "B")),
        ):
            values, mask = conditioning_query(present, stimulus_names=STIMULUS_NAMES)
            prob = cold.conditional_stimulus_probabilities(cold.state, values, mask)[STIMULUS_NAMES.index("US")]
            query_strings.append(f"{label}:{prob:.3f}")
        return (
            f"iter={iteration} cold_C={cold.state.num_causes} "
            f"accept={np.mean([t['accepted'] for t in cold.trace]) if cold.trace else 0.0:.3f} "
            f"swap={self.swap_accepts / max(1, self.swap_attempts):.3f} "
            f"logpost={cold.current_log_posterior:.2f} "
            f"queries=[{' '.join(query_strings)}] "
            f"edges={cold.state.edge_mask.astype(int).tolist()}"
        )

    def run(
        self,
        iterations: int,
        burn_in: int = 0,
        thin: int = 1,
        store_latents: bool = True,
        verbose_every: int = 0,
    ) -> list[PosteriorSample]:
        self.samples = []
        self.trace = []
        for iteration in range(1, iterations + 1):
            for sampler in self.samplers:
                sampler.step()
            swap_accepts = self.attempt_swaps()
            cold = self.cold_sampler
            record = {
                "iteration": iteration,
                "cold_num_causes": cold.state.num_causes,
                "cold_log_posterior": cold.current_log_posterior,
                "swap_accepts": swap_accepts,
                "swap_rate": self.swap_accepts / max(1, self.swap_attempts),
            }
            self.trace.append(record)

            should_store = iteration > burn_in and ((iteration - burn_in) % thin == 0)
            if should_store:
                latent_states = cold.sample_latent_states(cold.state) if store_latents else np.zeros(
                    (cold.num_trials, cold.state.num_causes),
                    dtype=np.int8,
                )
                self.samples.append(
                    PosteriorSample(
                        num_causes=cold.state.num_causes,
                        edge_mask=cold.state.edge_mask.copy(),
                        weights=cold.state.weights.copy(),
                        latent_bias=cold.state.latent_bias.copy(),
                        stimulus_bias=cold.state.stimulus_bias.copy(),
                        latent_states=latent_states,
                        log_posterior=float(cold.current_log_posterior),
                    )
                )
            if verbose_every and iteration % verbose_every == 0:
                print(self.verbose_summary(iteration))
        return self.samples


def sample_summary(samples: list[PosteriorSample]) -> dict[str, object]:
    if not samples:
        return {"num_samples": 0}

    cause_counts = np.asarray([sample.num_causes for sample in samples], dtype=int)
    unique, counts = np.unique(cause_counts, return_counts=True)
    return {
        "num_samples": len(samples),
        "mean_num_causes": float(np.mean(cause_counts)),
        "cause_count_pmf": {int(k): float(v / len(samples)) for k, v in zip(unique, counts)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RJMCMC reimplementation of Courville et al. (2003).")
    parser.add_argument("--ax-trials", type=int, default=4, help="Number of unreinforced A-X trials.")
    parser.add_argument("--iterations", type=int, default=4000, help="Total RJMCMC iterations.")
    parser.add_argument("--burn-in", type=int, default=1000, help="Burn-in iterations.")
    parser.add_argument("--thin", type=int, default=10, help="Posterior thinning interval.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--init-causes", type=int, default=1, help="Initial number of latent causes.")
    parser.add_argument("--verbose-every", type=int, default=0, help="Print sampler progress every N iterations.")
    args = parser.parse_args()

    data = generate_second_order_conditioning(ax_trials=args.ax_trials)
    sampler = RJMCMCSigmoidBeliefNetwork(data, init_causes=args.init_causes, seed=args.seed)
    samples = sampler.run(
        iterations=args.iterations,
        burn_in=args.burn_in,
        thin=args.thin,
        store_latents=True,
        verbose_every=args.verbose_every,
    )

    summary = sample_summary(samples)
    predictive_queries = {}
    for label, present in (
        ("P(US|A,D)", ("A",)),
        ("P(US|X,D)", ("X",)),
        ("P(US|B,D)", ("B",)),
        ("P(US|X,B,D)", ("X", "B")),
    ):
        values, mask = conditioning_query(present, stimulus_names=STIMULUS_NAMES)
        probs = sampler.posterior_predictive(values, mask, samples=samples)
        predictive_queries[label] = float(probs[STIMULUS_NAMES.index("US")])

    output = {
        "dataset": {
            "num_trials": int(data.shape[0]),
            "stimuli": list(STIMULUS_NAMES),
            "ax_trials": args.ax_trials,
        },
        "posterior": summary,
        "predictive_queries": predictive_queries,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
