# The Machine Learning Toolbox

This is a repository containing python utilities for machine learning and deep learning. 

## 1. Rule-Based Machine Learning: *RuleAugmentedEstimator Class*

**Combine domain knowledge facts with machine learning.**

This RuleAugmentedEstimator class wraps around sklearn estimators and augments functionality by adding rule-based logic.

The RuleAugmentedEstimator class works by allowing the user to add hard-coded rules which override model behavior.
Wherever available, the model wrapper will use hard-coded rules while making remaining predictions using the provided base machine learning model.
