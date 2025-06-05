# Olistchris subtitle
Data Engineering and Machine Learning Development on Brazilian E-Commerce Dataset by Olist.

# Git hub link:
https://github.com/Gary1E2/olistchris

# Olistchris Description
Olist is a Brazilian e-commerce marketplace like Lazada, Taobao and Shopee, it is a sales
platform that connects small retailers with customers.

The objective is to help Olist leverage data analytics and machine learning, focusing their business objective to build their customer base and increase future sales revenue.

### Repeat Buyers
Olist can drive profitable sales by stategically determining high value loyal customers to gain an enriching and comprehensive insight to their behaviour for improved decision making.

### Freight Value Mean
Olist can expand and retain their customer base by providing estimated added price for shipment, offering additional funtionality and improve user's trust and expectations.

### Delivery Time Mean
Olist can expand and retain their customer base by providing estimated delivery time for shipment, offering additional funtionality and improve user's trust and expectations.

### Conclusion
Olist can design and deploy more effective and targeted marketing campaigns for repeat customers. Trust and expectations can be built upon users by offering reliable, additional funtionalities.

# Changes to add after Presentation:
- containerizing run.sh
- README.md changes

# Contributors:
Chua Wen Hung Gary | 1e2.gary.chua@gmail.com

Ng Xin | ngxinramos@gmail.com

Arthur Stanley Wren | 230778hnyp@gmail.com

Keagan Lee | zeroediv@gmail.com

# Project Folder Overview:

## Project Folder Structure:
unused files/folders kept for future project expansion

# olistchris
- conf: configuration files
    - base : store catalog and pipeline parameters
        - catalog.yml : data registration (datasets, models, metrics)
        - parameters_data_processing.yml : unused folder
        - parameters_data_science.yml : store model parameters
        - parameters_reporting.yml : unused folder
        - parameters.yml : unused folder
    - ...

- data : data stored in different folders
    - 01_raw : raw datasets
    - 02_intermediate : processed datasets
    - 03_primary : final/model dataset
    - 04_feature : unused folder
    - 05_model_input : unused folder
    - 06_models : trained models
    - 07_model_output : unused folder
    - 08_reporting : evaluated model plots

- saved_models : saved models in .pickle format
- src : source code for processing, training, evaluation and reporting
    - olistchris : project name folder
        - pipelines : all 3 pipelines 
            - data_processing : data processing pipeline
            - data_science : machine learning pipeline
            - reporting : model reporting pipeline

        pipeline_registry.py : code for finding and registering pipelines
        ...

Dockerfile : for running a jupyter lab server container
eda.ipynb : ipynb file for eda
models.ipynb : ipynb file for model building
README.md : project context file (this file)
requirements.txt : project dependencies for installing
run.sh : runnable file for full pipeline execution

## Project Programming Language: 
### Python 3.12.7
## Project OS: 
### Windows 11

## Project Environment: 
- Kedro = 0.19.13
- Git = 2.49.0.windows.1
- Conda = 24.9.2

## Project Libraries for .ipynb:
- Pandas = 2.2.2
- Numpy = 1.26.4
- Matplotlib = 3.10.3
- Seaborn = 0.13.2
- Folium = 0.19.6
- Scikit-Learn = 1.5.1

## Project Libraries for pipeline:
- ipython >= 8.10
- jupyterlab >= 3.0
- kedro-datasets[pandas-csvdataset, pandas-exceldataset, pandas-parquetdataset, plotly-plotlydataset plotly-jsondataset, matplotlib-matplotlibwriter, tracking] >= 3.0
- kedro-viz >= 6.7.0
- kedro[jupyter] ~= 0.19.13
- notebook
- scikit-learn ~= 1.5.1
- seaborn ~= 0.13.1

# Overview of EDA:
The EDA provided useful insights into repeat buyer classification and two regression targets: freight value and delivery time. 

The repeat buyer target classes shows inbalance, some features showing noticeable differences between buyer types. The regression targets, both freight value and delivery time had skewed distributions, with a small number of extreme values. Several features suggest meaningful relationships with each target allowing for potential prediction.

# Pipeline Instructions:
## Pipeline Execution:
Pipelines are sequential and must be run from data_processing to data_science to reporting if relevant data is not stored. E.g model_input_table

## Pipeline Parameters:
Only the parameters_data_science.yml and parameters_reporting.yml has parameters that can be changed from .yml. Other parameters are only accessible in the pipeline nodes themselves and changes may result in errors or issues.

### Parameters_data_science.yml structure:
- model with nested parameters
    - features: features to train on
    - target: target to predict
    - classify: 1 for classfier model, 0 for regressor model (may break if target does not correspond to model type.)
    - <other model building parameters>: free to be adjusted within same data type
- model 2
    - ...
- ...

### Parameters_reporting.yml structure:
- plot type 1
    - fig_size: dimensions of the plot
    - plot specific parameters
    - ...
- plot type 2
    - ...
- ...

## Execute Commands:
- pip install -r requirements.txt
- kedro run (run all pipelines)
- kedro run --pipeline <pipeline> (run singular pipeline)

Example for model pipeline:
kedro run --pipeline data_science

## Docker Desktop Container Running:
### Jupyter Lab
- docker build -f JupyterLabDocker -t my-jupyter-app .
- docker run -p 8888:8888 my-jupyter-app
Go to http://localhost:8888/ 

### run.sh Pipeline Execution
- docker build -f PipelineDocker -t kedro-pipeline .
- docker run --rm kedro-pipeline

# Pipeline Flow:
There are 3 pipelines, data_processing, data_science and reporting. 

The provided datasets are ingested into the data_processing pipeline where only the orders dataset is processed in the preprocess_orders node. The preprocessed orders dataset and the raw datasets are then merged in the create_model_input_table_node where additional features are engineered before returning a final dataset. The processed orders dataset and the final dataset are both stored in the data folder under 02_intermediate and 03_primary respectively.

The final dataset is fed into the data_science pipeline where it is split, selecting only relevant features for predicting a target feature. The split data is used for training before being evaluated using multiple metrics, depending on model type. The metrics are printed out as logs when the nodes are running. This is repeated 2 more times for the next 2 models, storing the models in a .pickle format in the data folder under 06_models.

The saved models and split data, which is saved in the memory are used to create graphical plots, representing their metrics and performances. The classifer model is used to create a confusion matrix and ROC AUC plot while the other two regressor models are used to create true vs predicted plots.

# Model Selection:
## Repeat Buyers Classifer: 
GradientBoostingClassifier with optimized parameters using RandomizedSearchCV

The standard base model decision tree lacked in performance even when using GridSearchCV when compared to the optimized GradientBoostingClassifier ensemble model. Hence, the final model was derived after due experimentation and evaluation.

## Freight Value Regressor: 
GradientBoostingRegressor with optimized parameters using RandomizedSearchCV

The standard base models such as linear regression and decision tree was inadequate in producing strong performance metrics. Using ensemble models like AdaBoostingRegressor and GradientBoostingRegressor provided higher performance metrics with GradientBoostingRegressor being the best when optimized. Hence, the final model was derived after due experimentation and evaluation.

## Delivery Time Regressor: 
GradientBoostingRegressor with optimized parameters using RandomizedSearchCV

The standard base models provided mediocre performance metrics which greatly warranted the use of ensemble models. However, the ensemble models also struggled to fit to the data and produced better but still overall, weak performance metrics. GradientBoostingRegressor proved to have better performance metrics but only performing slightly better than predicting the target's mean value. Hence, the final model was derived after due experimentation and evaluation.

# Model Evaluation:
Both testing and training data was used to evaluate the model, allowing a gauge of every model's overfitting, variance or bias.

## Repeat Buyer Classifer:
### Test 
Accuracy: 0.9939

Precision: 0.9939

Recall: 0.9939

F1 Score: 0.9939

Specificity: 0.9967

ROC AUC Area: 0.9999

### Train 
Accuracy: 0.9998

Precision: 0.9998

Recall: 0.9998

F1 Score: 0.9998

Specificity: 0.9998

ROC AUC Area: 1.0000


Accuracy was used as the standard metric along with other metrics depending on confusion matrix evaluation. These metrics include Precision, Recall, F1 score, Specificity and the Confusion Matrix itself for a comprehensive evaluation of the model's prediction tendencies and biasness. Receiver Operatin Characteristics as well as the Area Under the Curve was used as the primary metric as the repeat buyer classes were significantly imbalanced, leaning heavily towards non-repeat buyers compared to repeat buyers. The ROC AUC plot allowed easy visual evaluation of the model as it reacts strongly to wrong predictions despite class imbalance.

## Freight Value Mean Regressor:
### Test 
MAE: -4.0805

MSE: -76.6362

R^2: 0.6735

### Train 
MAE: -3.6017

MSE: -65.5677

R^2: 0.7388

Mean Absolute Error was used as the primary metric as it was resistant to outliers allowing a clearer evaluation of the model's errors. This is followed by R Squared/R^2 which shows how well the model's predictions fit the true values, -1 being poor, 0 being equal to mean prediction and 1 being perfectly fitting the true values. Mean Squared Error is used to evaluate the model's supceptibility to outliers, increasing the error by a significant amount to highlight it.

## Delivery Time Mean Regressor:
### Test 
MAE: -4.8699

MSE: -71.6436

R^2: 0.1927

### Train 
MAE: -4.7454

MSE: -70.5480

R^2: 0.2128

The same metrics are used here although there is more emphasis on R Squared/R^2 with how poor the model was performing.

# Model Considerations:
Tree based models are relatively unaffected by feature scaling. This allows a simpler data processing stage where skewed data and targets are not required to be scaled before model training.

Ensemble models allows multiple trees to combine their predictions to produce a robust model that generalizes better with improved performance metrics.

RandomizedSearchCV is eventually used over GridSearchCV due to time constraints as RandomizedSearchCV is faster for model training.
