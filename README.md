# The Machine Learning Toolbox

This is a repository containing python utilities for machine learning and deep learning. 

## 1. Rule-Based Machine Learning: *RuleAugmentedEstimator Class*

**Combine domain knowledge facts with machine learning.**

This class is a wrapper class for sklearn estimators with the additional possibility of adding rule-based logic to the underlying estimator.
The provided rules are hard-coded and take precedence over the underlying estimator's predictions.
    
**Example Usage:**

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()

rules = {"House Type": [
            ("=", "Penthouse", 1.0),
            ("=", "Shack", 0.0)
         ],
         "House Price": [
            ("<", 1000.0, 0.0),
            (">=", 500000.0, 0.0)
         ]}
         
ra_estimator = RuleAugmentedEstimator(gbr, rules)
```
In the above example, whenever in th `House Type` pandas column the value "Penthouse" appears, the value `1.0` is returned by the estimator.
Similarily, if any value in the `House Price` pandas column is greater or equal to `500000.0`, the value `0.0` will be returned.
