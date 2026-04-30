import joblib
import xgboost as xgb
import numpy as np
import shap

m = joblib.load('app/ml/models/xgb_calibrated.pkl')
b = m.calibrated_classifiers_[0].estimator.get_booster()
X = np.zeros((1, 14))
fn = ["age", "gender", "glucose", "hemoglobin", "cholesterol", "bp_systolic", "bp_diastolic", "pulse_pressure", "tsh", "vitamin_d", "creatinine", "ldl", "hdl", "crp"]
import pandas as pd
X_df = pd.DataFrame(X, columns=fn)

try:
    ex = shap.TreeExplainer(b)
    shap_vals = ex.shap_values(X)
    print("SHAP with Numpy Array: SUCCESS")
except Exception as e:
    print("SHAP with Numpy Array failed:", repr(e))

try:
    ex = shap.TreeExplainer(b)
    shap_vals = ex.shap_values(X_df)
    print("SHAP with DataFrame: SUCCESS")
except Exception as e:
    print("SHAP with DataFrame failed:", repr(e))
