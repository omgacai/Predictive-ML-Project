import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from data_loader import SQLiteLoader
from type_stage_merger import PlantTypeStageMerger
from data_cleaning import data_cleaning_pipeline
from preprocess import preprocessor
from model import regression_models, classification_models
from config import CONFIG


def load_and_clean_data(db_path):
    """Load and clean data from SQLite database."""
    pipeline = Pipeline(steps=[
        ('loader', SQLiteLoader(db_path=db_path)),
        ('cleaner', data_cleaning_pipeline)
    ])
    return pipeline.fit_transform(None)


def prepare_regression_data(data, test_size, random_state, target_col='temperature_sensor_(°c)'):
    """Prepare regression data (features and target)."""
    X = data.drop(columns=[target_col])
    y = data[target_col]
    return train_test_split(X, y, test_size = test_size, random_state = random_state)


def train_regression_model(X_train, X_test, y_train, y_test):
    """Train regression pipeline and evaluate performance."""
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', regression_models)
    ])
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    y_pred = pipeline.predict(X_test)
    temp_score = pipeline.score(X_test, y_test)
    
    print(f"Temperature Prediction Model R^2 score: {temp_score:.4f}")
    return pipeline


def prepare_classification_data(data, test_size, random_state):
    """Prepare classification data by merging plant type and stage."""
    pipeline = Pipeline(steps=[('merger', PlantTypeStageMerger())])
    transformed_data = pipeline.fit_transform(data)

    y = transformed_data['plant_type_stage']
    X = transformed_data.drop(columns=['plant_type_stage', 'plant_type', 'plant_stage'])

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return train_test_split(X, y, test_size = test_size, random_state = random_state)


def train_classification_model(X_train, X_test, y_train, y_test):
    """Train classification pipeline and evaluate performance."""
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', classification_models)
    ])
    pipeline.fit(X_train, y_train)

    # Predictions & Evaluation
    #y_pred = pipeline.predict(X_test)
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)

    print(f"Plant Type-Stage Cross-validation scores: {scores}")
    return pipeline


def main():

    """Main function to run regression and classification models."""
    print("\nLoading and cleaning data...")
    clean_data = load_and_clean_data(CONFIG["data"]["db_path"])

    print("\nRunning Temperature Regression Model...")
    (X_train_temp, X_test_temp, y_train_temp, y_test_temp) = prepare_regression_data(clean_data, CONFIG["data"]["test_size"], CONFIG["data"]["random_state"])
    train_regression_model(X_train_temp, X_test_temp, y_train_temp, y_test_temp)

    print("\nRunning Plant Type-Stage Classification Model...\n")
    (X_train_stage, X_test_stage, y_train_stage, y_test_stage) = prepare_classification_data(clean_data, CONFIG["data"]["test_size"], CONFIG["data"]["random_state"])
    train_classification_model(X_train_stage, X_test_stage, y_train_stage, y_test_stage)


if __name__ == "__main__":
    main()
