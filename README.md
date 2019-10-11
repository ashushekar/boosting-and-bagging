# Boosting and Bagging

## Boosting
Boosting is a sequential technique which works on the principle of an ensemble. It combines a set of weak learners and
delivers improved prediction accuracy. At any instance t, the model outcomes are weighted based on the outcomes of 
previous instance t-1. The outcomes predicted correctly are given a lower weight and the ones miss-classified are 
weighted higher. Note that a weak learner is one which is slightly better than random guessing. For example, a decision
tree whose predictions are slightly better than 50%.