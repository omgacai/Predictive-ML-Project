from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

#regression models
regressor_model = RandomForestRegressor(n_estimators=100, random_state=42)


#classifier models
classifier_model = RandomForestClassifier(n_estimators=50, min_samples_split=4, random_state=42)

'''
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.linear_model import LinearRegression, LogisticRegression

#regression models
regression_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }


#classification models
classification_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=0)
    }


'''