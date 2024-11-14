from xgboost import XGBClassifier
import pandas as pd
import numpy as np

class XGBoostPreferenceModel(XGBClassifier):

    def __init__(self, **kwargs):
        super(XGBoostPreferenceModel, self).__init__(**kwargs)
        
    def explain(self, X: np.array, y: np.array, raw_data: pd.DataFrame) -> pd.Series:
        # XGBoost stores feature importances in 'feature_importances_' after fitting
        importances = self.feature_importances_
        return pd.Series(importances, index=raw_data.columns)
        
    def fit(self, X, y, *args, **kwargs):
        super(XGBoostPreferenceModel, self).fit(X, y)
        return self