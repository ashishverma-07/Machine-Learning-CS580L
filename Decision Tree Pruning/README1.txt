
Compilation instructions

Since my solution is based on Python there is no need for compiling it. You
can call it directly using the Python interpreter as
    
    python program.py <L> <K> <training-set> <validation-set> <test-set> <to-print>
	
	L: integer (used in the post-pruning algorithm)
	K: integer (used in the post-pruning algorithm)
	to-print:{yes,no}

Also, you can run the program directly using the bash script:

    ./program <L> <K> <training-set> <validation-set> <test-set> <to-print>
