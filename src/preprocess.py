from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


numerical_cols = [ 'humidity_sensor_(%)',
       'light_intensity_sensor_(lux)', 'co2_sensor_(ppm)', 'ec_sensor_(ds/m)',
       'o2_sensor_(ppm)', 'nutrient_n_sensor_(ppm)', 'nutrient_p_sensor_(ppm)',
       'nutrient_k_sensor_(ppm)', 'ph_sensor', 'water_level_sensor_(mm)',
      
]
categorical_cols = ['system_location_code', 'previous_cycle_plant_type']


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)
