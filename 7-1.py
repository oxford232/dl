import pandas as pd

pd.set_option("display.max_columns", None)

df = pd.read_csv(r"./titanic/train.csv")

validation_df = df.sample(n=100, random_state=42)

train_df = df.drop(validation_df.index)


file_path = "./titanic/validation_data2.csv"
train_file_path = "./titanic/train2.csv"

validation_df.to_csv(file_path, index=False)
train_df.to_csv(train_file_path, index=False)


# df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# df = df.dropna(subset=["Age"])

# df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)

# print(df.head(10))


