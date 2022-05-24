import pickle
import numpy as np
from tensorflow import keras
import pandas as pd
from ruamel.yaml import YAML
yaml = YAML(typ='safe')
import sys
import shutil

with open('params.yaml', 'rb') as params_file:
    model_params = yaml.load(params_file)
train_params = model_params.get("train", {})
predict_params = model_params.get("predict", {})

if not predict_params.get("predict", True):
    print("prediction disabled, exiting")
    shutil.copy("sample_submission.csv", "submission.csv")
    sys.exit(0)

model_files = []
if train_params.get("ensemble", False):
    for i in range(5):
        model_files.append("model/fold_" + str(i))
else:
    model_files.append("model")

with open('test.pickle', 'rb') as data_file:
    (X_test) = pickle.load(data_file)

preds = []
for f in model_files:
    print("running", f)
    fold_model = keras.models.load_model(f)
    fold_preds = fold_model.predict(X_test)
    preds.append(fold_preds)

n_preds = len(preds)
if n_preds > 1:
    preds = np.sum(preds, axis=0) / n_preds
else:
    preds = preds[0]

submission = pd.read_csv("sample_submission.csv")
submission["target"] = preds
submission.to_csv("submission.csv", index=False)