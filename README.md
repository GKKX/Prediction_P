## Usage
The software makes the trained XGBoost model for predicting permeability coefficient into an interactive desktop application for the user's convenience. Users do not need to install other ancillary software, open the main_Pred_P.exe can be used.

## This software has two functions: 
1- To calculate the gas molecular permeability coefficient in single-crystal materials.
   A single prediction result is displayed on the interface.
2- Batch computing permeability coefficient of crystal materials
   The predicted result will be saved in Result/Batch_Predicted_P.xlsx.

## This folder includes five folders:
1- Code
     1. XGBoost.py that has the code for the machine learning using XGBoost (for more info please visit: https://xgboost.readthedocs.io/en/stable/python/index.html)
     2. Prediction_P_code.py that has the code for a human-computer interactive interface software.

2- Extrapolation_data
     Example_C2H6.xlsx that is a sample file for batch prediction of material permeability.

3- Img 
     full_name.tif and sample_file.tif that are the interactive interface software required in the illustration picture. 
 
4- model
     xgboost.pt that is a trained XGBoost algorithm model.

5- Result 
     The predicted result will be automatically generated and saved in Result/Batch_Predicted_P.xlsx.



