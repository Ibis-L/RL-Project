import numpy as np
import pytest

from autoclassify.baselines.policies import (BestOnSmall, SuccessiveHalving, avg_rank_policy, exhaustive_policy,
                                             fixed_policy, random_policy)
from autoclassify.env.autoclassify_env import AutoClassifyEnv
from autoclassify.env.compat import HAVE_GYM
from autoclassify.pipeline import run_episode
from conftest import CLFS, LEVELS


def env_for(tiny, **kw):
    tables, context, keys = tiny
    return AutoClassifyEnv(tables, context, CLFS, LEVELS, keys[:4], **kw)


def test_spaces_and_mask(tiny):
    env = env_for(tiny)
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,) and env.action_space.n == 3 * 4 + 1
    assert info["action_mask"].all()                          # everything valid at start
    env.step(env.action(1, 2))                                # classifier b at 50%
    m = env.action_mask()
    assert not m[env.action(1, 0)] and not m[env.action(1, 2)] and m[env.action(1, 3)]
    with pytest.raises(ValueError):
        env.step(env.action(1, 1))                            # masked action is refused


def test_budget_never_exceeded_random_rollouts(tiny):
    env = env_for(tiny, seed=1)
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, info = env.reset()
        done = False
        while not done:
            a = int(rng.choice(np.flatnonzero(info["action_mask"])))
            obs, r, term, trunc, info = env.step(a)
            done = term
        assert info["budget_used"] <= env.budget + 1e-9
        assert 0.0 <= info["regret_test"] <= 1.0


def test_exhaustive_costs_one_and_regret_matches_hand_calculation(tiny):
    tables, context, keys = tiny
    env = env_for(tiny)
    k = keys[0]
    info = run_episode(env, exhaustive_policy(), k, None)
    assert np.isclose(info["budget_used"], 1.0)
    t = tables[k]
    s = int(np.argmax(t.val[:, -1]))                          # selected on validation
    q = t.test[:, -1]
    expected = 0.0 if q.max() == q.min() else (q.max() - q[s]) / (q.max() - q.min())
    assert np.isclose(info["regret_test"], expected)
    assert info["selected"] == CLFS[s]


def test_stop_immediately_gives_regret_one(tiny):
    env = env_for(tiny)
    env.reset(options={"instance": tiny[2][0], "budget": 0.5})
    _, r, term, _, info = env.step(env.STOP)
    assert term and info["regret_val"] == 1.0 and r == 0.0


def test_cutoff_spends_budget_and_returns_no_score(tiny):
    env = env_for(tiny)
    env.reset(options={"instance": tiny[2][0], "budget": 0.05})
    _, _, term, _, info = env.step(env.action(2, 3))          # most expensive fit, does not fit
    assert term and info["cutoff"] and np.isclose(info["budget_used"], 0.05) and info["selected"] is None


def test_all_baselines_respect_budget(tiny):
    env = env_for(tiny)
    pols = [random_policy(0), avg_rank_policy([2, 0, 1]), BestOnSmall([2, 0, 1]), SuccessiveHalving([2, 0, 1])]
    for p in pols:
        for B in (0.1, 0.25, 0.5, 1.0):
            info = run_episode(env, p, tiny[2][1], B)
            assert info["budget_used"] <= B + 1e-9, p.name
    info = run_episode(env, fixed_policy("b"), tiny[2][1], None)
    assert info["n_evals"] == 1


def test_flags_zero_blocks(tiny):
    env = env_for(tiny, flags={"features": False, "obs": False})
    obs, _ = env.reset(seed=0)
    env.step(env.action(0, 1))
    obs = env._obs()
    assert np.allclose(obs[-4:], 0)          # dataset features hidden
    assert np.allclose(obs[3:6], 0)          # observed scores hidden


def test_cost_seen_scales_to_full_budget(tiny):
    env = env_for(tiny)
    env.reset(options={"instance": tiny[2][0], "budget": None})
    env.step(env.action(1, 1))               # classifier b at 25%
    seen = env._obs()[6:9]
    assert np.isclose(seen[1], env.table.cost[1, 1] / 0.25) and seen[0] == 0 and seen[2] == 0


@pytest.mark.skipif(not HAVE_GYM, reason="gymnasium not installed")
def test_gymnasium_check_env(tiny):
    from gymnasium.utils.env_checker import check_env
    check_env(env_for(tiny), skip_render_check=True)
