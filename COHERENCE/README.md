The COHERENCY directory conains the files used to evaluate the accuracy, coherence, and contra-positive coherence of the GPT, Mistral, and Claude models.
It contains:
- this README.md file
- my Leap-of-Thought data set (data_hypernyms_hypernyms_explicit_only_short_neg_hypernym_rule_test.jsonl).
- test_consistency.py, to run tests of all the models on the LOT tuples.  Results are stored in the Results sub-directory.
- crunch_and_plot_results.py, which reads the files of responses (created by test_consistency.py), calculates accuracy and consistency, and plots the results.
- Results subdirectory, which contains:
     - the raw responses from all the models. Files are named after the dataset + model name
     - results_allModels_explicitOnlyTestData_withFilteringOnKnownEntailment, a text file with the calculated results for each model (accuracy, consistency, etc.)
     - with_logitbias, a sub-sub-directory, which contains:
             - the raw responses and calculated results when the GPT models were run with the logitbias forcing True/False.


