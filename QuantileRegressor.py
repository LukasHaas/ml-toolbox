import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import r2_score

from decimal import Decimal
from typing import Tuple, Union, Dict

class QuantileRegressor(BaseEstimator):
    """Predicts quantiles for regression problems.
    This class is a wrapper class for sklearn estimators with the additional
    possibility of adding rule-based logic to the underlying estimator.
    The provided rules are hard-coded and take precedence over the underlying
    estimator's predictions.
    """

    def __init__(self, base_regressor: BaseEstimator, fit_quantiles: list = None, quantile_range: Tuple = None, step: float = None, **base_params):
        """Initializes the QuantileRegressor instance.
        Initializes the quantile regressor by supplying the underlying
        sklearn estimator as well as fixed quantiles or a quantile range
        and step size.
        Args:
            base_regressor: The underlying sklearn estimator.
              Must implement a fit and predict method as well as accept loss and alpha parameters.
            fit_quantiles, optional: List of quantiles on which the model should be trained on.
              If no list is provided, the model falls back on the quantile_range and step parameters.
            quantile_range, optional: Tuple with a lower and higher quantile bound which 
              provide a range for quantiles on which the model should be trained on.
            step, optional: Step size which is used to create the model quantile range.
            **base_params: Optional keyword arguments which will be passed on
            to the ``base_model``.
        Examples:
            The below example illustrates how an instance of the 
            QuantileRegressor class can be initialized with a trained 
            sklearn GradientBoostingRegressor instance.
            >>> gbr = GradientBoostingRegressor()
            >>> quantile_reg = QuantileRegressor(gbr, fit_quantiles=[0.4, 0.5, 0.55])
        """

        assert {'loss', 'alpha'}.issubset(base_regressor.get_params().keys()), \
                'Provided base_regressor instance doesn\'t accept quantile loss function.'

        assert fit_quantiles is not None or (quantile_range is not None and step is not None), \
                'The variable fit_quantiles or the variables quantile_range and step must be specified.'
            
        params = {'loss': 'quantile', 
                  'alpha': 0.5}
        
        base_regressor = clone(base_regressor)
        base_regressor.set_params(**base_params)
        base_regressor.set_params(**params)
    
        self.base_regressor = base_regressor
        self.fit_quantiles = fit_quantiles
        self.quantile_range = quantile_range
        self.step = step
        
        model_dict = {}
        self._quantiles = [0.5]
        model_dict['0.5'] = base_regressor            
        self.model_dict = model_dict

        quantiles = self.__quantile_creator() if fit_quantiles is None and quantile_range is not None and step is not None \
                                             else fit_quantiles

        all_models = [self._create_model_from_quantile(q) for q in quantiles]
        
        for i in range(0, len(quantiles)):
            if quantiles[i] not in self.model_dict.keys():
                self.model_dict['{}'.format(quantiles[i])] = all_models[i]
            
        quantiles = self._quantiles + quantiles
        quantiles = list(set(quantiles))
        self._quantiles = sorted(quantiles)


    def __repr__(self):
        return "Quantile Regressor:\n\n\t Base Model: {}\n\t Quantiles: {}".format(self.base_regressor, self.get_quantiles())

    def __str__(self):
         return self.__str__
    
    def _create_model_from_quantile(self, quantile: float) -> BaseEstimator:
        """Returns a new copy of the base_regressor with the specified quantile.
        Args:
            quantile: The quantile to predict.
        Returns
            BaseEstimator: The new model with the adjusted quantile loss function.
        """
        
        new_model = sklearn.base.clone(self.base_regressor)
        
        params = {'loss': 'quantile', 
                  'alpha': quantile}
        
        new_model.set_params(**params)
        return new_model
    
    def __quantile_creator(self) -> list:
        """Returns a list of quantiles based on the specified range and step.
        Returns
            list: The quantiles to use for model training.
        """
        
        quantiles = []
        
        low = self.quantile_range[0]
        high = self.quantile_range[1]
        
        while low < high + self.step:
            quantiles.append(low)
            low = low + self.step
            
        decimal_step = Decimal('{}'.format(self.step))
        decimal_places = abs(decimal_step.as_tuple().exponent)
        quantiles = [round(x, decimal_places) for x in quantiles]
    
        return quantiles
        
        
    def fit(self, X: pd.DataFrame, y:pd.Series, additional_quantiles: list = [], **kwargs):
        """Fits the estimator to the data.
        
        Fits the estimator to the data, by creating copies of the underlying 
        estimator, each trained on a specific quantile.
        
        Args:
            X: The training feature data.
            y: The training label data.
            additional_quantiles, optional: List of additional quantiles on which 
              the model should be trained on.
            **kwargs: Optional keyword arguments passed to the underlying
              estimator's fit function.
        """
        
        new_quantiles = list(filter(lambda x: x not in self._quantiles, additional_quantiles))
        self._quantiles = sorted(self._quantiles + new_quantiles)
        
        new_models = [self._create_model_from_quantile(q) for q in new_quantiles]
            
        for i in range(0, len(new_quantiles)):
            quantile = new_quantiles[i]
            model = new_models[i]
            
            self.model_dict['{}'.format(quantile)] = model
        
        for quantile, model in self.model_dict.items():
            model.fit(X, y)

    def predict(self, X: Union[pd.DataFrame, pd.Series, np.array, list], quantiles: list = [0.5]) -> Union[Dict, np.array]:
        """Gets predictions for the provided input data and quantiles.
        
        The predicitons are evaluated for every quantile and input data 
        point using the underlying estimator.
        
        Args:
            X: The feature data to evaluate predictions for.
            quantiles, optional: The quantiles which should be estimated.
        
        Returns:
            Union[Dict, np.array]: The evaluated predictions.
              If multiple quantiles were provided, the result is a dictionary
              where the keys are the quantiles and the values the predictions.
        """

        quantile_set = set(quantiles)
        model_quantiles = set(self._quantiles)
        
        assert quantile_set.issubset(model_quantiles), \
            'The model hasn\'t been trained on all provided quantiles yet.\n\
            See get_quantiles() method for available quantile values or fit the model on additional quantiles.'
        
        model_predictions = {}
        
        for quantile in quantiles:
            
            model = self.model_dict['{}'.format(quantile)]
            predictions = model.predict(X)
            
            model_predictions['{}'.format(quantile)] = predictions
            
        if len(quantiles) == 1:
            return list(model_predictions.values())[0]
        
        else:
            return model_predictions

    def predict_confidence_interval(self, X: Union[pd.DataFrame, pd.Series, np.array, list], confidence: float) -> list:
        """Predicts confidence intervals for the provided input data and quantiles.
        
        The model must first be trained on the quantiles resulting from
        the confidence level, meaning 0.5 +- (confidence / 2).
        
        Args:
            X: The feature data to evaluate predictions for.
            confidence: The confidence for which the interval
              should be calculated (0.0 - 1.0)
        
        Returns:
            list: The evaluated confidence intervals.
              The return type is a list of tuples which each show the confidence
              interval range.
        """

        assert confidence >= 0 and confidence <= 1, \
            "The confidence must be between 0.0 and 1.0"

        first_quantile = 0.5 - (confidence / 2)
        second_quantile = 0.5 + (confidence / 2)

        quantile_set = set([first_quantile, second_quantile])
        model_quantiles = set(self._quantiles)

        assert quantile_set.issubset(model_quantiles), \
            'The model hasn\'t been trained yet on the needed quantiles for the {} % confidence interval.\n\
             Please fit the model first on all of the following quantiles: {}.'.format(int(confidence * 100), quantile_set)

        predictions = self.predict(X, quantiles=[first_quantile, second_quantile])
        pred_keys = list(predictions.keys())
        data_length = len(predictions[pred_keys[0]])

        confidence_intervals = [(predictions[pred_keys[0]][x], predictions[pred_keys[1]][x]) for x in range(data_length)]
        return confidence_intervals
    
    def score(self, X: Union[pd.DataFrame, pd.Series, np.array, list], y: Union[pd.Series, np.array, list], quantile=0.5) -> float:
        """Evaluates the r2 score for the predictions.
        
        Evaluates the r2 score for the predictions at the specified quantile.
        
        Args:
            X: The feature data to evaluate the r2 score for.
            y: The true values at the specified quantile.
            quantile, optional: The quantile at which to score the estimator.
              In most cases, only the value 0.5 makes sense for scoring.
        
        Returns:
            float: The r2 score.
        """
        
        predictions = self.predict(X, quantiles=[quantile])
        score = r2_score(y, predictions)
        
        return score
    
    def get_params(self, deep: bool = True) -> Dict:
        """Return the model's and base regressor's parameters.
        Args:
            deep: Whether to recursively return the base model's parameters.
        Returns
            Dict: The model's parameters.
        """
        
        params = {'base_regressor': self.base_regressor,
                  'fit_quantiles': self.fit_quantiles,
                  'quantile_range': self.quantile_range,
                  'step': self.step,
                  'model_dict': self.model_dict}
        
        params.update(self.base_regressor.get_params(deep=deep))
        return params
    
    def set_params(self, **params):
        """Sets parameters for the model and base regressor.
        Args:
            **params: Optional keyword arguments.
        """
        
        parameters = params.copy()
        
        if 'base_regressor' in parameters:
            value = parameters.pop('base_regressor')
            self.base_regressor = value
            
        if 'fit_quantiles' in parameters:
            value = parameters.pop('fit_quantiles')
            self.fit_quantiles = value
            
        if 'quantile_range' in parameters:
            value = parameters.pop('quantile_range')
            self.quantile_range = value
            
        if 'step' in parameters:
            value = parameters.pop('step')
            self.step = value
            
        if 'outcome_range' in parameters:
            value = parameters.pop('outcome_range')
            self.outcome_range = value
            
        if 'model_dict' in parameters:
            value = parameters.pop('model_dict')
            self.model_dict = value
        
        self.base_regressor.set_params(**parameters)
        
        for model in self.model_dict.values():
            model.set_params(**parameters)
            
    def get_quantiles(self) -> list:
        """Returns the quantiles on which the model was trained.
        Returns:
            list: The quantiles on which the model was trained on.
        """
        return self._quantiles
