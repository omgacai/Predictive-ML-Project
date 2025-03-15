import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from data_loader import SQLiteLoader
from type_stage_merger import PlantTypeStageMerger
from data_cleaning import data_cleaning_pipeline
from preprocess import regression_preprocessor, classification_preprocessor
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
    """Train regression models and evaluate performance with GridSearchCV."""

    print(f"Training and evaluating model: RandomForestRegressor")

    # Extract param_grid from the config file
    param_grid = CONFIG['regression_param_grid']

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', regression_preprocessor),  # Your preprocessor here
        ('model', regression_models)
    ])

    # Use GridSearchCV to search for the best parameters
    grid_search = GridSearchCV(pipeline, param_grid, scoring='r2', n_jobs=-1, verbose=1)
    
    
    # Fit the model with GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Get the best model and its performance
    best_model = grid_search.best_estimator_

    # Evaluation
    y_pred = best_model.predict(X_test)
    temp_score = best_model.score(X_test, y_test)

    print(f"Best Model: {grid_search.best_params_}")
    print(f"Best Model R^2 score on Test Data: {temp_score:.4f}")
    
    return best_model


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
    """Train multiple classification models and evaluate performance using train-test validation."""

    for model_name, model in classification_models.items():
        print(f"Training {model_name}...")

        # Create a pipeline for each model
        pipeline = Pipeline(steps=[
            ('preprocessor', classification_preprocessor),  # Replace with your preprocessor
            ('model', model)  # Use the model from the dictionary
        ])
        
        # Fit the pipeline on the training data
        pipeline.fit(X_train, y_train)
        
        # Make predictions on the test data
        y_pred = pipeline.predict(X_test)
        
        # Evaluate performance on the test set
        accuracy = accuracy_score(y_test, y_pred)
        
        # Print the results
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}\n")
    
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
