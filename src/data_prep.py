import os
import pandas as pd

from sklearn.model_selection import train_test_split

DATA_PATH = "data/tourism.csv"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Remove unnecessary index and customer identifier
df.drop(
    columns=["Unnamed: 0", "CustomerID"],
    errors="ignore",
    inplace=True
)

TARGET = "ProdTaken"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

if df[TARGET].isna().any():
    raise ValueError("Target column contains missing values.")

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("ProdTaken must contain binary labels 0 and 1.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_df = X_train.copy()
train_df[TARGET] = y_train

test_df = X_test.copy()
test_df[TARGET] = y_test

train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

print("Data preparation completed.")
print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)
print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))
print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))
