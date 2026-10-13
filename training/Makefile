.PHONY: fix lint test
CHECK = uv run --no-project --python 3.12 --with-requirements requirements-check.txt

fix:
	$(CHECK) ruff format .
	$(CHECK) ruff check --fix .
lint:
	$(CHECK) ruff check .
	$(CHECK) ruff format --check .
test:
	PYTHONPATH=src:../policy/src $(CHECK) pytest -q
