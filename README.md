The code is fully self-contained. To run the 10 tests, simply vary the SEED value from 0 to 9 (main, line 32).

The results of the tests are stored in results -> batch_all_instances.
A text file is generated for each instance and stored in the folder corresponding to its size. A text file summarizing each 
instance size is also created.
Finally, a JSON file containing additional information for all runs is created.

WARNING: Between two consecutive runs, the files stored in batch_all_instances are deleted to make room for new results. If
you wish to keep the results, it is therefore important to move the results elsewhere between two consecutive runs.
