import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class ColumnRenamer(BaseEstimator, TransformerMixin):
    """Standardizes column names (lowercase, replaces spaces with underscores)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df = df.rename(lambda x: x.lower().strip().replace(' ', '_'), axis='columns')
        return df
    
class FilterInvalid(BaseEstimator, TransformerMixin):
    """Converts numeric columns and removes invalid values."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Remove rows where light intensity is invalid
        df = df[df['light_intensity_sensor_(lux)'] >= 0]
        
        # Filter out invalid temperature values (less than 0 degrees)
        df = df[df['temperature_sensor_(°c)'] >= 0]
        return df


class StandardiseCategoryNames(BaseEstimator, TransformerMixin):
    """Formats text fields by capitalizing words."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """for both plant type and plant stage"""
        df = X.copy()
        df['plant_type'] = df['plant_type'].str.title()
        df['plant_stage'] = df['plant_stage'].str.title()
        return df


class NumericConverter(BaseEstimator, TransformerMixin):
    """Converts numeric columns and removes invalid values."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        num_cols = ['nutrient_n_sensor_(ppm)', 'nutrient_p_sensor_(ppm)', 'nutrient_k_sensor_(ppm)']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df



class FeatureCreator(BaseEstimator, TransformerMixin):
    """Creates new features like plant_type_stage and categorizes plant_stage."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Create a new feature combining plant_type and plant_stage
       
        # Convert plant_stage to a categorical feature with ordered categories
        category_order = ['Seedling', 'Vegetative', 'Maturity']
        df['plant_stage'] = pd.Categorical(df['plant_stage'], categories=category_order, ordered=True)

        return df


# Combine all steps into a data cleaning pipeline
data_cleaning_pipeline = Pipeline(steps=[
    ('rename_columns', ColumnRenamer()),
     ('remove_invalid_samples', FilterInvalid()),
    ('standardise_cat_names', StandardiseCategoryNames()),
    ('convert_numeric', NumericConverter()),
    ('create_features', FeatureCreator())
])

