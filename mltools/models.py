import lightgbm as lgb
import numpy as np

def build_gbm_model(X_train, y_train, X_test, y_test, gbm_params, objective='reg', early_stopping_rounds=500, error='mape', verbose=500):
    if objective == 'reg':
        model = lgb.LGBMRegressor
    elif objective == 'classif':
        model = lgb.LGBMClassifier
    else:
        raise AttributeError('Unknow objective')
    gbm = model(**gbm_params)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=error,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds
            )
    return gbm


def get_error_gbm(error_func, transform=None):
    def gbm_error_func(y_true, y_pred):
        if transform == 'log':
            y_pred = np.exp(y_pred)
            y_true = np.exp(y_true)
        eval_result = error_func(y_true, y_pred)
        return 'error', eval_result, False
    return gbm_error_func