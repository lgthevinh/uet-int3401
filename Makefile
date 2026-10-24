.PHONY: grade

Q?="q3"

grade:
	@echo "Run autograder"
	
	python autograder.py -q $(Q) 
