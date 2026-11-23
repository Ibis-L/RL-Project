"""AutoClassify: reinforcement learning for budget-aware algorithm selection in classification.

An RL agent, trained across many datasets, decides step by step which classifier to evaluate at
which training-data budget (Continue / Switch) and when to STOP. Its state holds simple dataset
features, the scores and costs it has observed so far, and the budget left. There is no separate
meta-learner: the policy itself learns which classifiers suit which data.
"""

__version__ = "1.1.0"
