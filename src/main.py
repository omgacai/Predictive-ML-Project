from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from data_loader import SQLiteLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

def prepare_regression_data(data, test_size, random_state, target_col):
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

    rand_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, 
                                    scoring='r2', n_iter=10, n_jobs=-1, 
                                    verbose=1, cv=5)
    
    rand_search.fit(X_train, y_train)
    
    # Get the best model and its performance
    best_model = rand_search.best_estimator_

    # Evaluation
    y_pred = best_model.predict(X_test)

    #scores
    r2_score = best_model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
  

    print(f"\nBest Model: {rand_search.best_params_}")
    print(f"Best Model R^2 score on Test Data: {r2_score:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")

    metrics = {
        'r2_score': r2_score,
        'mae': mae,
        'mse': mse
    }
    
    return best_model, metrics

def prepare_classification_data(data, test_size, random_state, target):
    """Prepare classification data by merging plant type and stage."""
    pipeline = Pipeline(steps=[('merger', PlantTypeStageMerger())])
    transformed_data = pipeline.fit_transform(data)

    y = transformed_data[target]
    X = transformed_data.drop(columns=[target])

    return train_test_split(X, y, test_size = test_size, random_state = random_state)


def train_classification_model(X_train, X_test, y_train, y_test):
    """Train multiple classification models with hyperparameter tuning using RandomizedSearchCV."""
    
    # Create a pipeline with a placeholder model
    pipeline = Pipeline(steps=[
        ('preprocessor', classification_preprocessor),  # Your preprocessing pipeline
        ('model', classification_models)  # Your base model (will be replaced during tuning)
    ])
    
    # Apply GridSearchCV
    grid_search = GridSearchCV(
        pipeline, 
        CONFIG["class_param_grid"], 
        cv=5,  # 5-fold cross-validation
        scoring='f1_weighted',  
        n_jobs=-1  # Use all available cores for speed
        
    )

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Get the best model from the search
    best_pipeline = grid_search.best_estimator_
    
    # Predict using the best model
    y_pred = best_pipeline.predict(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print the results
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    
    # Print confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Print detailed classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Store metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    return best_pipeline, metrics, grid_search.best_params_


def main():

    """Main function to run regression and classification models."""
    print("\nLoading and cleaning data...")
    clean_data = load_and_clean_data(CONFIG["data"]["db_path"])

    print("\nRunning Temperature Regression Model...\n")
    (X_train_temp, X_test_temp, y_train_temp, y_test_temp) = prepare_regression_data(clean_data, CONFIG["data"]["test_size"], CONFIG["data"]["random_state"], CONFIG["regression"]["target"])
    train_regression_model(X_train_temp, X_test_temp, y_train_temp, y_test_temp)

    print("\nRunning Plant Type-Stage Classification Model...\n")
    (X_train_stage, X_test_stage, y_train_stage, y_test_stage) = prepare_classification_data(clean_data, CONFIG["data"]["test_size"], CONFIG["data"]["random_state"], CONFIG["classification"]["target"])
    train_classification_model(X_train_stage, X_test_stage, y_train_stage, y_test_stage)


if __name__ == "__main__":
    main()
