from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from data_loader import SQLiteLoader
from type_stage_merger import PlantTypeStageMerger
from data_cleaning import data_cleaning_pipeline
from preprocess import preprocessor
from model import regressor_model, classifier_model


def main():
    # 1. load and clean the data first
    clean_pipeline = Pipeline(steps=[
        ('loader', SQLiteLoader(db_path='data/agri.db')),
        ('cleaner', data_cleaning_pipeline),
        ('merger', PlantTypeStageMerger())
    ])

    clean_data = clean_pipeline.fit_transform(None)
    
    # Train-test split for regression (temperature)
    X = clean_data.drop('temperature_sensor_(°c)', axis=1)  
    y_temp = clean_data['temperature_sensor_(°c)']
    y_temp = y_temp.fillna(y_temp.median())
    
    #Train-test split for regression (temperature)
    X_train, X_test, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=0)


    y_stage = clean_data['plant_type_stage']
    # Train-test split for classification (plant type-stage)
    X_train_stage, X_test_stage, y_train_stage, y_test_stage = train_test_split(X, y_stage, test_size=0.2, random_state=0)

    regression_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', regressor_model)  # Your regression model
    ])
    
    # Plant type-stage classification model pipeline (classification)
    classification_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', classifier_model)  # Your classification model
    ])

    regression_pipeline.fit(X_train, y_train_temp)
    
    # Train the classification model (plant type-stage categorization)
    classification_pipeline.fit(X_train_stage, y_train_stage)
    
    # Make predictions for temperature (regression)
    y_pred_temp = regression_pipeline.predict(X_test)
    
    # Make predictions for plant type-stage (classification)
    y_pred_stage = classification_pipeline.predict(X_test_stage)
    
    # Calculate the R^2 score for the regression model (temperature prediction)
    temp_score = regression_pipeline.score(X_test, y_test_temp)
    
    # Calculate accuracy score for the classification model (plant type-stage prediction)
    stage_score = accuracy_score(y_test_stage, y_pred_stage)
    
    # Print the results
    print(f"Temperature Prediction Model R^2 score: {temp_score}")
    print(f"Plant Type-Stage Classification Model Accuracy: {stage_score}")
    
    return regression_pipeline, classification_pipeline, temp_score, stage_score, X_train, X_test, y_train_temp, y_test_temp, y_train_stage, y_test_stage


if __name__ == "__main__":
    main()