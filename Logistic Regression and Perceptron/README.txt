
Compilation instructions

Since my solution is based on Python there is no need for compiling it. You
can call it directly using the Python interpreter as
    
    python2 Logistic.py <train-ham-set> <train-spam-set> <test-ham-set> <test-spam-set>
	
For example:
    python2 Logistic.py train/ham/ train/spam/ test/ham/ test/spam/

Also, you can run the program directly using the bash script:

    ./program <train-ham-set> <train-spam-set> <test-ham-set> <test-spam-set>
