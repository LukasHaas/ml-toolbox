import pandas as pd
import sklearn

from sklearn.base import BaseEstimator

class RuleAugmentedEstimator(BaseEstimator):
    """Augments sklearn estimators with rule-based logic.

    This class is a wrapper class for sklearn estimators with the additional
    possibility of adding rule-based logic to the underlying estimator.
    The provided rules are hard-coded and take precedence over the underlying
    estimator's predictions.

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, base_estimator, rules, **base_params):
        """Initializes the RuleAugmentedEstimator instance.

        Initializes the rule-augmented estimator by supplying the underlying
        sklearn estimator as well as the hard-coded rules.

        Args:
            base_estimator (BaseEstimator): The underlying sklearn estimator.
              Must implement a fit and fit_transform method.
            rules (dict): The hard-coded rules in the format of a dictionary,
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
            to the ``base_estimator``.

        Examples:
            The below example illustrates how an instance of the 
            RuleAugmentedClass can be initialized with a trained sklearn
            GradientBoostingRegressor instance.

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

        self.base_estimator = base_estimator
        self.rules = rules

        assert isinstance(self.base_estimator, BaseEstimator)
        assert isinstance(self.rules, dict)

        self.base_estimator.set_params(**base_params)


    def __repr__(self):
        return "Rule Augmented Estimator:\n\n\t Base Model: {}\n\t Rules: {}".format(self.base_estimator, self.rules)

    def __str__(self):
         return self.__str__
   
    def __reduce_to_necessary_features__(self, X, y):
        """Reduces the trainig data to data points not affected by the rules."""
        
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

    def fit(self, X, y, **kwargs):
        """Fits the estimator to the data.
        
        Fits the estimator to the data, only training the underlying estimator
        on data which isn't affected by the hard-coded rules.
        
        Args:
            X (pandas.DataFrame): the training feature data.
            y (pandas.DataFrame/pandas.Series): the training label data.
            **kwargs: Optional keyword arguments passed to the underlying
            estimator's fit function.
            
    """
        train_x, train_y = self.__reduce_to_necessary_features__(X, y)
        self.base_estimator.fit(train_x, train_y, **kwargs)
    
    def predict(self, X):
        """Gets predictions for the provided feature data.
        
        The predicitons are evaluated using the provided rules wherever possible
        otherwise the underlying estimator is used.
        
        Args:
            X (pandas.DataFrame): The feature data to evaluate predictions for.
        
        Returns:
            np.array: Evaluated predictions.
        """
        
        predict_X = X.copy()
        predict_X['prediction'] = np.nan

        for category, rules in self.rules.items():

            if category not in predict_X.columns.values: continue

            for rule in rules:

                if rule[0] == "=":
                    predict_X.loc[predict_X[category] == rule[1], 'prediction'] = rule[2]

                elif rule[0] == "<":
                    predict_X.loc[predict_X[category] < rule[1], 'prediction'] = rule[2]

                elif rule[0] == ">":
                    predict_X.loc[predict_X[category] > rule[1], 'prediction'] = rule[2]

                elif rule[0] == "<=":
                    predict_X.loc[predict_X[category] <= rule[1], 'prediction'] = rule[2]

                elif rule[0] == ">=":
                    predict_X.loc[predict_X[category] >= rule[1], 'prediction'] = rule[2]

                else:
                    print("Invalid rule detected: {}".format(rule))

        if len(predict_X.loc[predict_X['prediction'].isna()].index != 0):

            base_X = predict_X.loc[predict_X['prediction'].isna()].copy()
            base_X.drop('prediction', axis=1, inplace=True)
            predict_X.loc[predict_X['prediction'].isna(), 'prediction'] = self.base_estimator.predict(base_X)

        return predict_X['prediction'].values
    
    def get_params(self, deep=True):
        """Gets parameters of the model.

        Args:
            deep (bool): If true, iterates recursively through model parameters.

        Returns:
            dict: Parameters.
        """
        
        params = {'base_estimator': self.base_estimator,
                  'outcome_range': self.outcome_range,
                  'rules': self.rules
                 }
    
        params.update(self.base_estimator.get_params(deep=deep))
        return params
    
    def set_params(self, **params):
        """Sets parameters of the model.

        Args:
            **params: Keyword arguments specifying the model parameters.
        """
                  
        parameters = params
        param_keys = parameters.keys()
        
        if 'base_estimator' in param_keys:
            value = parameters.pop('base_estimator')
            self.base_estimator = value
            
        if 'rules' in param_keys:
            value = parameters.pop('rules')
            self.rules = value
        
        self.base_estimator.set_params(**parameters)