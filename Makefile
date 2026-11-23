# Convenience targets. Usage: make smoke   |   make stage CONFIG=configs/stage1.yaml
CONFIG ?= configs/stage1.yaml
PY ?= python

.PHONY: install test smoke stage fetch split perfdb status features train evaluate ablate figures test-final

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

smoke:
	$(PY) -m autoclassify.cli all --config configs/smoke.yaml

stage: fetch split perfdb status features train evaluate figures

fetch split perfdb status features train evaluate ablate figures:
	$(PY) -m autoclassify.cli $@ --config $(CONFIG)

# Opens meta-test ONCE (refuses a second time unless you add --force)
test-final:
	$(PY) -m autoclassify.cli evaluate --config $(CONFIG) --split test
	$(PY) -m autoclassify.cli figures --config $(CONFIG) --split test
