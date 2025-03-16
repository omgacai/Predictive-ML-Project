# aiip5-Yong-Sook-Mun-124H
### Full name: Yong Sook Mun
### Email address: yongsm1223@gmail.com


# b. Folder Structure 
```plaintext
├── data/                             # Not pushed to git
│   ├── agri.db                       # The raw dataset used for analysis (Not pushed to git)
├── eda.ipynb                         # Exploratory Data Analysis (EDA) Jupyter Notebook
├── src/
│   ├── data_loader.py                # Load data
│   ├── data_cleaning.py              # Clean data
│   ├── type_stage_merger.py          # Form and preprocess new column plant_type_stage for the classification model
│   ├── preprocess.py                 # Preprocessing data 
│   ├── model.py                      # Loading models 
│   ├── main.py                       # Main script to train and test the model
│   ├── config.py                     # Load config.yaml 
├── config.yaml                       # Configuration file with hyperparameters and model parameters
├── README.md                        
├── run.sh                      
└── requirements.txt
```
# c. Instructions for executing pipeline
## Prerequisites
Before running the pipeline, ensure that the required libraries are installed. You can install the dependencies using `pip`:
```
pip install -r requirements.txt
```

## Running the pipeline
1. **Data Preperation:** create a folder 'data' and place the raw dataset, calls.db inside (relative path), modify the path in config.yaml if needed 
2. **Preprocessing and Feature Engineering:** Default parameters are set but you can modify them in the config.yaml file
> Please note that the maximum number of iterations for the regression model is set to 10. Therefore, if the parameters you input result in fewer than 10 combinations, a warning will be displayed but nothing else will be affected.
3. Executing model: head into the main folder directory and run this in your terminal
```
run.sh
```
# d. Pipeline flow 
[same for the 2 models]
1. **Data Loading:** Loading the data from the file
2. **Data Cleaning:** Cleaned the dataframe
- standardised column names to snake_case
- filtered invalid rows (negative ```light_intensity_sensor_(lux)``` and negative ```temperature_sensor_(°c)```)
- standardised the capitalisation of category names in ```plant_type``` and ```plant_stage```
- converted the 3 nutrients, ```nutrient_n_sensor_(ppm), nutrient_p_sensor_(ppm), nutrient_k_sensor_(ppm)``` from object to numeric type
- ordered the categories in ```plant_stage``` in this order: ['Seedling', 'Vegetative', 'Maturity']

[different for the 2 models]

3. **Data Preprocessing:** Steps taken to clean and prepare the data.
4. **Model Training:** Algorithms used and training procedures.
5. **Model Evaluation:** Metrics and evaluation techniques.

| **Step**             | **Regression**                                                                                                                                                                                                                         | **Classification**                                                                                                                                                                                                                      |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data Preprocessing** | - **Handling Missing Values:** Impute missing numerical values using mean, median, or mode. <br> - **Encoding Categorical Variables:** Convert categories to numerical values using techniques like one-hot encoding or label encoding. <br> - **Feature Scaling:** Standardize features to ensure they contribute equally to the model's performance. <br>  - **Splitting Data:** Divide the dataset into training and testing subsets, commonly using an 80/20 split. | - **Handling Missing Values:** Impute missing categorical values using the most frequent category. <br> - **Encoding Categorical Variables:** Apply one-hot encoding or label encoding to transform categories into numerical format.  <br> - **Splitting Data:** Partition the dataset into training and testing sets, typically using an 80/20 or 70/30 ratio. |
| **Model Training**   | - **Algorithm Selection:** Choose appropriate regression algorithms such as Linear Regression or Random Forest <br> - **Model Fitting:** Train the model on the training dataset to learn the relationships between features and the continuous target variable. <br> - **Hyperparameter Tuning:** Optimize model parameters to enhance performance, often using techniques like Grid Search or Randomized Search with cross-validation. | - **Algorithm Selection:** Opt for suitable classification algorithms like Logistic Regression or Random Forests. <br> - **Model Fitting:** Train the model on the training data to learn the patterns distinguishing different classes. <br> - **Hyperparameter Tuning:** Adjust model parameters to improve accuracy, utilizing methods like Grid Search or Randomized Search with cross-validation. |
| **Model Evaluation** | - **Performance Metrics:** Assess model accuracy using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or R-squared.  | - **Performance Metrics:** Evaluate model effectiveness using metrics like Accuracy, Precision, Recall, F1 Score, and Area Under the ROC Curve (AUC-ROC). |


# e. Overview of key findings from EDA
1. The dataset is generally well-balanced, with no major imbalances between categories. Missing data were addressed by removing invalid temperature and light intensity readings, while the remaining numerical missing values were imputed with median values. This ensures that the analysis isn't biased by outliers or gaps in the data.

2. Upon examining the relationship between variables and plant type/stage, it became evident that a combination of multiple variables is necessary to make accurate predictions. No single variable on its own could sufficiently explain the variations across plant stages. For instance, the box plot revealed that only 1-2 types/stages stood out for each factor, with others remaining quite similar.

3. The analysis of temperature and light intensity suggests a more complex relationship. While temperature showed weak correlations with other factors, scatter plots of temperature versus light intensity revealed distinct clustering patterns, hinting at a more interesting trend that could potentially be useful for predictions.

# f. How features are preprocessed
| **Preprocessing Step**       | **Description**                                                                                                                                                                                                                          |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Handling Missing Values**  | Numerical: impute with median, Categorical: impute with most frequent (not applicable here as all the categorical variables do not have missing values) |
| **Encoding Categorical Data**| Converting categorical variables into numerical formats to facilitate model processing. Techniques include:  |
|                              | - **One-Hot Encoding:** Creates binary columns for each category. (other predictors)|
|                              | - **Label Encoding:** Assigns unique integers to categories. (for target variable) |
| **Feature Scaling**          | Standardizing the range of independent variables to ensure uniformity. Methods include:  |
|                              | - **Min-Max Scaling:** Scales data to a fixed range, typically [0, 1]. |
|                              | - **Z-Score Normalization (Standardization):** Centers data around 0 with a standard deviation of 1. |
| **Data Splitting**          | Dividing the dataset into training, validation, and test sets to evaluate model performance effectively. I used the 80/20 ratio |

# g/h. Choice of models & elaboration
## ML Task 1: Predicting Temperature conditions 
### Chosen Model: Random Forest
> Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mean prediction (regression) of the individual trees. This model is particularly effective in this regression task of temperature prediction due to its ability to handle complex, non-linear relationships and interactions between features. (As shown from the EDA, no other numerical features share a strong correlation with temperature) Moreover, I compared the results with varying parameters of other models and Random Forest achieves the highest R² values and lowest mean absolute errors. 

### Evaluation
To fine-tune the hyperparameters, I first used a RandomizedSearchCV with certain parameters (can be edited in config.yaml). To narrow down the window I changed to using GridSearchCV on parameters close to the one obtained from RandomizedSearchCV. The best hyperparameters obtained with RandomForestRegression() is:

| **Parameter**                     | **Value**    | **Explanation**                                       |
|------------------------------------|--------------|-------------------------------------------------------|
| Number of Trees                   | 300          | The number of trees in the Random Forest model.       |
| Max Depth of Each Tree            | None         | Trees are allowed to grow without a maximum depth limit. |
| **R² Score**                       | 0.7012       | Indicates the proportion of variance explained by the model; higher is better. |
| **Mean Absolute Error (MAE)**     | 0.6649       | The average of the absolute differences between predicted and actual values. |
| **Mean Squared Error (MSE)**      | 0.7828       | Measures the average squared difference between predicted and actual values, penalizing larger errors more. |

> The model demonstrates a solid performance with an R² score of 0.7012, suggesting that it explains a good portion of the variance in the data. The error metrics, including Mean Absolute Error (MAE) of 0.6649 and Mean Squared Error (MSE) of 0.7828, indicate that the model's predictions are reasonably accurate, with a balanced trade-off between precision and error penalization. Overall, the model is quite decent and can be considered for further fine-tuning or deployment depending on the specific use case.

## ML Task 2: Categorising Plant Type-Stage
### Chosen Model: Random Forest
> Before testing, random forest stood out as similar to the reason in ML task 1, it is effective in handling complex and non-linear relationships between features. Moreover, I tested out other models like Logistic Regression and KNeighborsClassifier which achieved a relatively lower accuracy score.

### Evaluation
Hence, after fine-tuning the random forest classification model using grid search, here are the best hyperparameters:

| **Parameter**                              | **Value**   | **Explanation**                                                       |
|--------------------------------------------|-------------|-----------------------------------------------------------------------|
| Number of Trees                            | 400         | The total number of decision trees in the Random Forest model.        |
| Max Depth of Each Tree                     | None        | The trees are allowed to grow without a maximum depth limit.          |
| Min Number of Samples to Split a Node      | 2           | The minimum number of samples required to split a node.               |
| **Accuracy**                               | 0.8530      | The proportion of correct predictions made by the model.              |
| **Precision**                              | 0.8547      | The ratio of correctly predicted positive observations to total predicted positives. |
| **Recall**                                 | 0.8530      | The ratio of correctly predicted positive observations to total actual positives.  |
| **F1 Score**                               | 0.8531      | The harmonic mean of precision and recall, providing a balance between the two. |

> The four performance metrics—accuracy, precision, recall, and F1 score—are all around 0.85, which is exceptionally high. These results indicate that the model is not only making accurate predictions, but also effectively identifying positive cases while minimizing false positives and false negatives. The close alignment of precision, recall, and F1 score further highlights the model's robustness and balanced performance, demonstrating its ability to handle the dataset with impressive reliability and consistency.

# i. Other considerations for deploying the models 
|Consideration|Elaboration|
|---|---|
|**1. Computational Efficiency in Hyperparameter Tuning**: |While tuning the parameters, while a grid search will be more comprehensive as it goes through all the possible combinations, it is computationally expensive, and takes a long time to run especially for the regression model. More specifically, regression models use continuous targets which takes a longer time to tune as compared to classification models with discrete targets. Hence, I used the Randomised Search to tune the regression model and Grid Search for the classification model.|
|**2. Cross-Validation Strategy**: |The dataset was found to be fairly balanced through EDA, making **random** cross-validation appropriate. I selected 5-fold cross-validation as the optimal balance between computational efficiency and statistical reliability. This provided sufficient validation of model performance without excessive runtime.|
|3. **Optimizing Number of Estimators:** |Through empirical testing, I observed diminishing returns in model performance beyond certain thresholds. Specifically, accuracy improvements became negligible after 300 estimators for regression and 400 for classification. This insight allowed me to select parameter values that maximize predictive power while minimizing computational overhead - an important consideration for eventual deployment.|
