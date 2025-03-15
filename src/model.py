from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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
        
    first_model_name = list(models.keys())[0] 
    return models[first_model_name]


regression_models = get_models(CONFIG["regression_models"], "regression")

classification_models = get_models(CONFIG["classification_models"], "classification")
