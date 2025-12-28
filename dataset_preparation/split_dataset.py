import pandas as pd
from sklearn.model_selection import train_test_split

metadata_csv = "tensor_metadata_maia23.csv"

df = pd.read_csv(metadata_csv)
train_val, test = train_test_split(df, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1111, random_state=42)  # ~10% val

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

train.to_csv("tensor_metadata_train3.csv", index=False)
val.to_csv("tensor_metadata_val3.csv", index=False)
test.to_csv("tensor_metadata_test3.csv", index=False)
print("Train, validation, and test splits saved to CSV files.")