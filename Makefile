.PHONY: grade demo

Q?="q5"

grade:
	@echo "Run autograder"

	python autograder.py -q $(Q)

demo:
	python scripts/demo/q3_loss_race.py
