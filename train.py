from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from sklearn.model_selection import train_test_split

from dvclive.keras import DvcLiveCallback
from ruamel.yaml import YAML
import pickle

# load the parameter file and select the 'train' section:
yaml = YAML(typ='safe')
with open('params.yaml', 'rb') as params_file:
    model_params = yaml.load(params_file)
params = model_params.get("train", {})
print("Parameters:", params)

# load the training/validation datasets
print("laoding dataset")
with open('train.pickle', 'rb') as data_file:
    (x_train, y_train) = pickle.load(data_file)
print(x_train.shape, y_train.shape)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)

def make_model():
    model = Sequential()
    model.add(Dense(512, input_dim=x_train.shape[1], activation="swish"))
    model.add(Dense(256, activation="swish"))
    model.add(Dense(128, activation="swish"))
    model.add(Dense(64, activation="swish"))
    model.add(Dense(1, activation="sigmoid"))
    return model

model = make_model()

opt = Adam(learning_rate = params.get("lr", 0.01))
lf = BinaryCrossentropy()
metrics = [AUC(name="auc")]
model.compile(optimizer=opt, loss=lf, metrics=metrics)
model.fit(
    x_train, y_train,
    validation_data=(x_valid, y_valid),
    batch_size = params.get("bs", 2048),
    epochs=params.get("epochs", 10),
    callbacks=[DvcLiveCallback(path="training_metrics")])

model.save("model")