import joblib
try:
    m = joblib.load('app/ml/models/xgb_calibrated.pkl')
    features = list(m.feature_names_in_) if hasattr(m, 'feature_names_in_') else list(m.calibrated_classifiers_[0].estimator.feature_names_in_)
    print("FEATURES:", features)
except Exception as e:
    print("ERROR:", e)
