.PHONY: grade

Q?="q5"

grade:
	@echo "Run autograder"
	
	python autograder.py -q $(Q) 
