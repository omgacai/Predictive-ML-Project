from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class PlantTypeStageMerger(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def _combine_features(self, X):
        return X['plant_type'].astype(str) + "_" + X['plant_stage'].astype(str)

    def fit(self, X, y=None):
        combined = self._combine_features(X)
        self.label_encoder.fit(combined)
        return self

    def transform(self, X):
        df = X.copy()
        combined = self._combine_features(df)
        df['plant_type_stage'] = self.label_encoder.transform(combined)
        df = df.drop(columns=['plant_type', 'plant_stage'])

        print("Labels Encoded: \n")
        for index, label in enumerate(self.label_encoder.classes_):
            print(f"Label {index}: {label}")
        return df
