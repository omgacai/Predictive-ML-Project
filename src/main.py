from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from data_loader import SQLiteLoader
from data_cleaning import data_cleaning_pipeline
from preprocess import preprocessor
from model import model


def main():
    # load and clean the data first
    clean_pipeline = Pipeline(steps=[
        ('loader', SQLiteLoader(db_path='data/agri.db')),
        ('cleaner', data_cleaning_pipeline)
    ])
    clean_data = clean_pipeline.fit_transform(None)
    
    # Perform train-test split on cleaned data
    X = clean_data.drop('temperature_sensor_(°c)', axis=1)  
    y = clean_data['temperature_sensor_(°c)']
    y = y.fillna(y.median())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # preproccessing and model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Fit on training data
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate on test data
    score = model_pipeline.score(X_test, y_test)
    print(f"Model score: {score}")
    

    return model_pipeline, score, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()