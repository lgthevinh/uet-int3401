.PHONY: grade

Q="q2"

grade:
	@echo "Run autograder"
	
	python autograder.py -q $(Q) 
