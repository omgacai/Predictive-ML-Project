from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from config import CONFIG


#loading models

def get_models(model_config, model_type):
    models = {}
    for model_name, params in model_config.items():
        model_class = globals().get(model_name)  
        if model_class:
            models[model_name] = model_class(**params)  
        else:
            raise ValueError(f"Unknown {model_type} model: {model_name}")
        
    return models


#regression_models = get_models(CONFIG["regression_models"], "regression")

classification_models = get_models(CONFIG["classification_models"], "classification")


model_params = CONFIG['regression_models']['RandomForestRegressor']
    
# Create the model with parameters from the config file
regression_models = RandomForestRegressor(**model_params)