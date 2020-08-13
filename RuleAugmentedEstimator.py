import numpy as np
import pandas as pd
import sklearn

from typing import Dict, Tuple
from sklearn.base import BaseEstimator

class RuleAugmentedEstimator(BaseEstimator):
    """Augments sklearn estimators with rule-based logic.
    This class is a wrapper class for sklearn estimators with the additional
    possibility of adding rule-based logic to the underlying estimator.
    The provided rules are hard-coded and take precedence over the underlying
    estimator's predictions.
    """

    def __init__(self, base_model: BaseEstimator, rules: Dict, **base_params):
        """Initializes the RuleAugmentedEstimator instance.
        Initializes the rule-augmented estimator by supplying the underlying
        sklearn estimator as well as the hard-coded rules.
        Args:
            base_model: The underlying sklearn estimator.
              Must implement a fit and predict method.
            rules: The hard-coded rules in the format of a dictionary,
              with keys being the pandas dataframe column name, and the values
              being a tuple in the following form:
              
              (comparison operator, value, return value)
              Acceptable comparison operators are:
              "=", "<", ">", "<=", ">="
              Example:
              
              {"House Type": [
                  ("=", "Penthouse", 1.0),
                  ("=", "Shack", 0.0)
               ],
               "House Price": [
                   ("<", 1000.0, 0.0),
                   (">=", 500000.0, 1.0)
              ]}
            **base_params: Optional keyword arguments which will be passed on
            to the ``base_model``.
        Examples:
            The below example illustrates how an instance of the 
            RuleAugmentedEstimator class can be initialized with a trained 
            sklearn GradientBoostingRegressor instance.
            >>> gbr = GradientBoostingRegressor()
            >>> rules = {"House Type": [
                            ("=", "Penthouse", 1.0),
                            ("=", "Shack", 0.0)
                         ],
                         "House Price": [
                            ("<", 1000.0, 0.0),
                            (">=", 500000.0, 1.0)
                        ]}
            >>> ra_estimator = RuleAugmentedEstimator(gbr, rules)
        """

        self.rules = rules
        self.base_model = base_model
        self.base_model.set_params(**base_params)

    def __repr__(self):
        return "Rule Augmented Estimator:\n\n\t Base Model: {}\n\t Rules: {}".format(self.base_model, self.rules)

    def __str__(self):
         return self.__str__
   
    def _get_base_model_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Filters the trainig data for data points not affected by the rules."""
        
        train_x = X

        for category, rules in self.rules.items():

            if category not in train_x.columns.values: continue

            for rule in rules:

                if rule[0] == "=":
                    train_x = train_x.loc[train_x[category] != rule[1]]

                elif rule[0] == "<":
                    train_x = train_x.loc[train_x[category] >= rule[1]]

                elif rule[0] == ">":
                    train_x = train_x.loc[train_x[category] <= rule[1]]

                elif rule[0] == "<=":
                    train_x = train_x.loc[train_x[category] > rule[1]]

                elif rule[0] == ">=":
                    train_x = train_x.loc[train_x[category] < rule[1]]

                else:
                    print("Invalid rule detected: {}".format(rule))
                
        indices = train_x.index.values
        train_y = y.iloc[indices]
        
        train_x = train_x.reset_index(drop=True)
        train_y = train_y.reset_index(drop=True)
        
        return train_x, train_y   

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fits the estimator to the data.
        
        Fits the estimator to the data, only training the underlying estimator
        on data which isn't affected by the hard-coded rules.
        
        Args:
            X: The training feature data.
            y: The training label data.
            **kwargs: Optional keyword arguments passed to the underlying
            estimator's fit function.
            
    """
        train_x, train_y = self._get_base_model_data(X, y)
        self.base_model.fit(train_x, train_y, **kwargs)
    
    def predict(self, X: pd.DataFrame) -> np.array:
        """Gets predictions for the provided feature data.
        
        The predicitons are evaluated using the provided rules wherever possible
        otherwise the underlying estimator is used.
        
        Args:
            X: The feature data to evaluate predictions for.
        
        Returns:
            np.array: Evaluated predictions.
        """
        
        p_X = X.copy()
        p_X['prediction'] = np.nan

        for category, rules in self.rules.items():

            if category not in p_X.columns.values: continue

            for rule in rules:

                if rule[0] == "=":
                    p_X.loc[p_X[category] == rule[1], 'prediction'] = rule[2]

                elif rule[0] == "<":
                    p_X.loc[p_X[category] < rule[1], 'prediction'] = rule[2]

                elif rule[0] == ">":
                    p_X.loc[p_X[category] > rule[1], 'prediction'] = rule[2]

                elif rule[0] == "<=":
                    p_X.loc[p_X[category] <= rule[1], 'prediction'] = rule[2]

                elif rule[0] == ">=":
                    p_X.loc[p_X[category] >= rule[1], 'prediction'] = rule[2]

                else:
                    print("Invalid rule detected: {}".format(rule))

        if len(p_X.loc[p_X['prediction'].isna()].index != 0):

            base_X = p_X.loc[p_X['prediction'].isna()].copy()
            base_X.drop('prediction', axis=1, inplace=True)
            p_X.loc[p_X['prediction'].isna(), 'prediction'] = self.base_model.predict(base_X)

        return p_X['prediction'].values
    
    def get_params(self, deep: bool = True) -> Dict:
        """Return the model's and base model's parameters.
        Args:
            deep: Whether to recursively return the base model's parameters.
        Returns
            Dict: The model's parameters.
        """
        
        params = {'base_model': self.base_model,
                  'outcome_range': self.outcome_range,
                  'rules': self.rules
                 }
    
        params.update(self.base_model.get_params(deep=deep))
        return params
    
    def set_params(self, **params):
        """Sets parameters for the model and base model.
        Args:
            **params: Optional keyword arguments.
        """
                  
        parameters = params
        param_keys = parameters.keys()
        
        if 'base_model' in param_keys:
            value = parameters.pop('base_model')
            self.base_model = value
            
        if 'rules' in param_keys:
            value = parameters.pop('rules')
            self.rules = value
        
        self.base_model.set_params(**parameters)
