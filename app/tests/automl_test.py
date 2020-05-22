import unittest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from ml.optimizers import AutoMlSearch
import autosklearn.regression
from tests.test_helpers import get_df


class AutoMlTests(unittest.TestCase):

    def test_auto_gen(self):
        frame = get_df('pollution_data.csv')
        cls = autosklearn.regression.AutoSklearnRegressor
        scoring = "r2"


if __name__ == "__main__":
    data = get_df('dengue-dataset.csv')
    cls = autosklearn.regression.AutoSklearnRegressor
    scoring = "r2"
    temp = '/home/tony/temp'
    out = '/home/tony/out'
    targetAttributes = ['cases']
    targets = data[targetAttributes[0]].as_matrix()
    targets = np.reshape(targets, (len(targets), len(targetAttributes)))
    data.drop(targetAttributes[0], axis=1, inplace=True)
    # data.drop(data.columns[[0]], axis=1, inplace=True)

    ftypes = data.dtypes
    X_train, X_test, y_train, y_test = train_test_split(data.as_matrix(), targets, test_size=0.25,
                                                        random_state=123)
    ftypes = list(map(lambda x: 'Categorical' if str(x) != 'float64' else 'Numerical', ftypes))
    aml = AutoMlSearch(autosklearn.regression.AutoSklearnRegressor, temp, out, ftypes, scoring)
    aml.fit(X_train, y_train)
    m = aml.best_estimator_
    cv = aml.cv_results_
    y_true, y_pred = y_test, aml.predict(X_test)
    y_pred_proba = aml.predict_proba(X_test)
    acc = mean_squared_error(y_true, y_pred)
    pass
