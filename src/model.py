from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


regressor_model = RandomForestRegressor(n_estimators=100, random_state=42)

classifier_model = RandomForestClassifier(n_estimators=100, random_state=42)