
import numpy as np
import pandas as pd

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encoder(train,
                   test,
                   target,
                   gb_features,
                   prior=None,
                   min_samples_leaf=1,
                   smoothing=1,
                   noise_level=0,
                   loo=False):
    """    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    """
    target_mean = train[target].mean()
    for feature in gb_features:
        if isinstance(feature, list):
            name = '_'.join(feature) + '_mean'
        else:
            name = feature + '_mean'

        if name in train.columns:
            train.drop(name, axis=1, inplace=True)
        if name in test.columns:
            test.drop(name, axis=1, inplace=True)

        # Compute target mean
        averages = train.groupby(by=feature)[target].agg(["mean", "count"])

        # Compute smoothing
        smoothing_vals = 1 / \
            (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

        # Apply average function to all target data
        if prior is None:
            prior_vals = target_mean
        else:
            prior_vals = train.groupby(by=feature)[prior].mean()
            prior_vals[prior_vals.isnull()] = target_mean

        # The bigger the count the less full_avg is taken into account
        averages[target] = prior_vals * \
            (1 - smoothing_vals) + averages["mean"] * smoothing_vals
        averages.drop(["mean", "count"], axis=1, inplace=True)

        # Apply averages to train and test series
        tmp = averages.reset_index().rename(columns={target: 'average'})

        train = pd.merge(train, tmp, on=feature, how='left').rename(
            columns={'average': name})

        count = train.groupby(by=feature)[target].transform("count")
        if loo:
            train[name] = (train[name]*count - train[target])/(1.0*(count - 1).replace(0, 1))
        train.loc[:, name] = add_noise(
            train.loc[:, name].fillna(target_mean), noise_level)

        train[name].clip(0, 1, inplace=True)
        # print(train[(train["BuySell"] == "Sell") & (train["CustomerIdx"] == 2779) & (train["IsinIdx"] == 4960)])
        test = pd.merge(test, tmp, on=feature, how='left').rename(
            columns={'average': name})
        test.loc[:, name] = add_noise(
            test.loc[:, name].fillna(target_mean), noise_level)

    return train, test