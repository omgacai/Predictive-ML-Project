from sklearn.base import BaseEstimator, TransformerMixin


class PlantTypeStageMerger(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        df['plant_type_stage'] = df['plant_type'].astype(str) + "_" + df['plant_stage'].astype(str)
        return df
    
