from sklearn.base import BaseEstimator


class SequenceClassifier(BaseEstimator):
    """Generic class which other classifiers should extend.
    It inherits from BaseEstimator from scikit-learn.

    Methods to implement: _fit and _predict.
    A classifier should also have a name in order to distinguish it from other classifiers.

    Other guidelines for creating custom classifier:
        * All arguments of __init__ should have default value and the same name as they will have as the attributes of created object.
        * No confirmation of input parameters or taking data as argument should be in __init__ method. That belongs to fit method.

    Source: http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
    """

    def __init__(self, name, transformer):
        self.name = name
        self.transformer = transformer

    def fit(self, X, y):
        if self.transformer:
            transformed_data = self.transformer.fit_transform(X)
            return self._fit(transformed_data, y)
        return self._fit(X, y)

    def _fit(self, X, y):
        """
        Fit method is responsible mainly for training model.

        It should also check the parameters.
        New attributes can be also added here to the object - they should be ended by _ at the end, e.g. `self.fitted_`.
        Fit method should return self (for compatibility reasons with common interface of scikit-learn).
        Source: http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/

        Parameters
        -----------
        X : features to train on
        y: vector of classes to train on, same length as X

        Returns
        -----------
        self: the classifier itself
        """
        raise NotImplementedError

    def predict(self, X):
        if self.transformer:
            transformed_data = self.transformer.transform(X)
            return self._predict(transformed_data)
        return self._predict(X)

    def _predict(self, X):
        """
        Predict method is responsible for predicting the results based on X.

        It should return predictions.

        Parameters
        -----------
        X : features to predict from

        Returns
        -----------
        y: vector of predicted classes, same length as X
        """
        raise NotImplementedError
