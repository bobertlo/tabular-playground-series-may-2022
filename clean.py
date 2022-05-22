import pandas as pd
import pickle

print("loading data ...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

def process_string(df, field):
    for i in range(10):
        df['ch_' + str(i)] = df[field].str.get(i).apply(ord) - ord('A')
    df["unique_characters"] = df[field].apply(lambda s: len(set(s)))
    return df

print("processing f_27 characters ...")
train_df = process_string(train_df, "f_27")
test_df = process_string(test_df, "f_27")

# drop id and f_27 and create final datasets
print("preparing final datasets ...")
x_train = train_df.drop(["id","target","f_27"], axis=1)
y_train = train_df["target"]
x_test = test_df.drop(["id", "f_27"], axis=1)

print("writing pickles ...")
with open('train.pickle', 'wb') as data_file:
    pickle.dump((x_train, y_train), data_file)
with open('test.pickle', 'wb') as data_file:
    pickle.dump(x_test, data_file)