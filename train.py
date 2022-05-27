from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from dvclive.keras import DvcLiveCallback
from ruamel.yaml import YAML
import pickle
import json
import os
import math

# load the parameter file and select the 'train' section:
yaml = YAML(typ='safe')
with open('params.yaml', 'rb') as params_file:
    model_params = yaml.load(params_file)
params = model_params.get("train", {})
print("Parameters:", params)

epochs = params.get("epochs")
lr_start = params.get("lr_start")
lr_end = params.get("lr_end")
ensemble = params.get("ensemble")
dropout = params.get("dropout")
batch_size = params.get("bs")
batch_norm = params.get("batch_norm")

def cosine_decay(epoch):
    if epochs > 1:
        w = (1 + math.cos(epoch / (epochs-1) * math.pi)) / 2
    else:
        w = 1
    return w * lr_start + (1 - w) * lr_end

# load the training/validation datasets
print("laoding dataset")
with open('train.pickle', 'rb') as data_file:
    (X_train, y_train) = pickle.load(data_file)
print(X_train.shape, y_train.shape)

def make_model():
    model = Sequential()
    model.add(Dense(512, 
        kernel_regularizer=regularizers.l2(1e-5),
        input_dim=X_train.shape[1], activation="swish"))
    if dropout > 0: model.add(Dropout(dropout))
    if batch_norm > 0: model.add(BatchNormalization())
    model.add(Dense(512, 
        kernel_regularizer=regularizers.l2(1e-5),
        activation="swish"))
    if dropout > 0: model.add(Dropout(dropout))
    if batch_norm > 1: model.add(BatchNormalization())
    model.add(Dense(256,
        kernel_regularizer=regularizers.l2(1e-5),
        activation="swish"))
    if dropout > 0: model.add(Dropout(dropout))
    if batch_norm > 2: model.add(BatchNormalization())
    model.add(Dense(128,
        kernel_regularizer=regularizers.l2(1e-5),
        activation="swish"))
    if dropout > 0: model.add(Dropout(dropout))
    if batch_norm > 3: model.add(BatchNormalization())
    model.add(Dense(1, activation="sigmoid"))
    return model


def train_model(X_train, X_valid, y_train, y_valid, callbacks=[]):
    scaler = None
    if params.get("scale", False):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.fit_transform(X_valid)

    model = make_model()
    opt = Adam(learning_rate = lr_start)
    lf = BinaryCrossentropy()
    metrics = [AUC(name="auc")]
    lr = LearningRateScheduler(cosine_decay)
    callbacks.append(lr)
    model.compile(optimizer=opt, loss=lf, metrics=metrics)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size = batch_size,
        epochs=epochs,
        callbacks=callbacks)
    metrics = {}
    metrics['auc'] = history.history['auc'][-1]
    metrics['val_auc'] = history.history['val_auc'][-1]
    metrics['loss'] = history.history['loss'][-1]
    metrics['val_loss'] = history.history['val_loss'][-1]

    return model, scaler, metrics

os.makedirs("scalers/", exist_ok=True)

scaler = None
if params.get("ensemble", False):
    kf = KFold(n_splits=5)
    metrics = []
    for fold, (idx_tr, idx_va) in enumerate(kf.split(X_train)):
        X_tr = X_train.iloc[idx_tr]
        X_va = X_train.iloc[idx_va]
        y_tr = y_train.iloc[idx_tr]
        y_va = y_train.iloc[idx_va]

        fold_model, fold_scaler, fold_metrics = train_model(X_tr, X_va, y_tr, y_va)
        fold_model.save("model/fold_" + str(fold))
        with open('scalers/scaler_%i.pickle' % fold, 'wb') as data_file:
            pickle.dump(fold_scaler, data_file)
        metrics.append(fold_metrics)
    mean_metrics = {}
    for key in metrics[0].keys():
        key_metrics = []
        for m in metrics:
            key_metrics.append(m[key])
        mean_metrics[key] = sum(key_metrics) / len(key_metrics)
    with open("training_metrics.json", "w") as json_file:
        json.dump(mean_metrics, json_file)
    # dvc is looking for this from dvclive in single run mode...
    os.makedirs("training_metrics/scalars", exist_ok=True)

else:
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train)
    model, scaler, _ = train_model(X_train, X_valid, y_train, y_valid, callbacks=[DvcLiveCallback(path="training_metrics")])
    model.save("model")
    
    with open('scalers/scaler.pickle', 'wb') as data_file:
        pickle.dump(scaler, data_file)