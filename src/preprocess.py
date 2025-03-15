from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder
from config import CONFIG


numerical_cols = [ 'humidity_sensor_(%)',
       'light_intensity_sensor_(lux)', 'co2_sensor_(ppm)', 'ec_sensor_(ds/m)',
       'o2_sensor_(ppm)', 'nutrient_n_sensor_(ppm)', 'nutrient_p_sensor_(ppm)',
       'nutrient_k_sensor_(ppm)', 'ph_sensor', 'water_level_sensor_(mm)',
      
]
categorical_cols = ['system_location_code', 'previous_cycle_plant_type']


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy= CONFIG["preprocessing"]["numerical"]["imputer"])),
    ('scaler', StandardScaler() if  CONFIG["preprocessing"]["numerical"]["scaler"] == "standard" else MinMaxScaler())
])

cat_strategy = CONFIG["preprocessing"]["categorical"]["imputer"]
cat_encoder = CONFIG["preprocessing"]["categorical"]["encoder"]

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy= cat_strategy )),
    ('encoder', OneHotEncoder(handle_unknown='ignore') 
        if cat_encoder == "onehot" 
        else LabelEncoder() if cat_encoder == "label" 
        else OrdinalEncoder())
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)
