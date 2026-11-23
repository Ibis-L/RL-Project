import numpy as np

from autoclassify.agents.dqn import DQNAgent, train_dqn
from autoclassify.agents.mlp import MLP
from autoclassify.agents.qlearning import QLearningAgent, train_qlearning
from autoclassify.baselines.policies import AgentPolicy
from autoclassify.env.autoclassify_env import AutoClassifyEnv
from autoclassify.pipeline import run_episode
from conftest import CLFS, LEVELS


def test_mlp_gradient_matches_finite_differences():
    net = MLP([4, 6, 3], seed=0)
    x = np.random.default_rng(0).normal(size=(5, 4))
    g_out = np.random.default_rng(1).normal(size=(5, 3))
    out, acts = net.forward(x, cache=True)
    gW, _ = net.backward(acts, g_out)
    eps, i, j = 1e-6, 1, 2
    net.W[0][i, j] += eps
    up = (net.forward(x) * g_out).sum()
    net.W[0][i, j] -= 2 * eps
    down = (net.forward(x) * g_out).sum()
    net.W[0][i, j] += eps
    assert np.isclose(gW[0][i, j], (up - down) / (2 * eps), atol=1e-5)


DQN_CFG = {"hidden": [32, 32], "lr": 1e-3, "gamma": 0.99, "batch_size": 32, "buffer_size": 5000, "warmup": 200,
           "target_update": 200, "train_steps": 3000, "eps_start": 1.0, "eps_end": 0.05, "eps_decay_frac": 0.5,
           "eval_every": 3000, "grad_clip": 10.0}


def test_ddqn_learns_better_than_stopping(tiny):
    tables, context, keys = tiny
    env = AutoClassifyEnv(tables, context, CLFS, LEVELS, keys[:4], seed=0)
    agent = DQNAgent(env.obs_dim, env.action_space.n, DQN_CFG, double=True, seed=0)
    train_dqn(env, agent)
    regrets = [run_episode(env, AgentPolicy(agent, "ddqn"), k, 0.5)["regret_val"] for k in keys[:4]]
    assert np.mean(regrets) < 0.5          # stopping at once would give regret 1


def test_qlearning_runs_and_backs_off(tiny):
    tables, context, keys = tiny
    env = AutoClassifyEnv(tables, context, CLFS, LEVELS, keys[:4], seed=0)
    cfg = {"episodes": 2000, "alpha": 0.2, "gamma": 1.0, "eps_start": 1.0, "eps_end": 0.05,
           "eps_decay_frac": 0.5, "eval_every": 2000, "use_data_bucket": True}
    agent = QLearningAgent(env.action_space.n, cfg, seed=0)
    train_qlearning(env, agent)
    assert len(agent.Q) > 0 and len(agent.Qc) <= len(agent.Q)
    unseen = (0, 3, 0, 0, 1, 99)                               # top-1 value never seen
    mask = np.ones(env.action_space.n, bool)
    assert 0 <= agent.act(unseen, mask) < env.action_space.n
