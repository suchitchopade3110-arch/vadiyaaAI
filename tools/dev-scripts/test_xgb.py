import joblib
import xgboost as xgb
import numpy as np

m = joblib.load('app/ml/models/xgb_calibrated.pkl')
b = m.calibrated_classifiers_[0].estimator.get_booster()
print('Booster feature_names:', b.feature_names)

X = np.zeros((1, 14))
try:
    d = xgb.DMatrix(X)
    b.predict(d)
    print("Predict with no names: SUCCESS")
except Exception as e:
    print("Predict with no names failed:", repr(e))

try:
    fn = ["age", "gender", "glucose", "hemoglobin", "cholesterol", "bp_systolic", "bp_diastolic", "pulse_pressure", "tsh", "vitamin_d", "creatinine", "ldl", "hdl", "crp"]
    d2 = xgb.DMatrix(X, feature_names=fn)
    b.predict(d2)
    print("Predict WITH names: SUCCESS")
except Exception as e:
    print("Predict WITH names failed:", repr(e))
