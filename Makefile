.PHONY: install train eval lint format

install:
	pip install -r requirements.txt

train:
	python -m src.train

eval:
	python -m src.eval

lint:
	ruff check src tests

format:
	ruff format src tests
