.PHONY: grade

Q="q1"

grade:
	@echo "Run autograder"
	
	python autograder.py -q $(Q) 
