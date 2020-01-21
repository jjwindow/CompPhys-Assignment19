COMPUTATIONAL PHYSICS ASSIGNMENT 2019
JACK J WINDOW
IMPERIAL COLLEGE LONDON

This assignment is entirely programmed in functional python. Python is a high level language that
can be easily read and interpreted and also has a large selection of libraries which allow for 
functions such as FFTs and random number generation which are essential in this assignment. Its
reader-friendliness aids debugging as well as marking and understanding by the assessor. Its biggest
downfall is the lack of an accessible extended precision float data type, which is supported in C,
however learning C would be untenable for the purpose of this assingment.

The code for this assignment is found in three files:

assignmentTools.py 	- a small library of functions used throughout the program 
		     	  of no major conceptual consequence, but useful for checking
		          and validation procedures etc. This will not need editing.
assignmentFunctions.py -  primary library of functions used to carry out the exercises,
			  organised by question. This file will not need editing or running 
			  to replicate results, but contains all of the functional insight.
assignmentRun.py       -  The code which executes the relevant functions for each question
			  in such a way as to produce the results obtained. It is advised to
			  select a question at a time and run selection, which is why "from
			  assignmentFunctions import *" is repeated at every question.

Happy Computing!