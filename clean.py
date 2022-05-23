import pandas as pd
import pickle

print("loading data ...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# def process_string(df, field):
for df in [train_df, test_df]:
    for i in range(10):
        df['ch_' + str(i)] = df["f_27"].str.get(i).apply(ord) - ord('A')
    df["unique_characters"] = df["f_27"].apply(lambda s: len(set(s)))
    df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
    df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)

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