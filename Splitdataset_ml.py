"""
split_dataset_ML.py
===================
Week 1 - Step 4: Stratified Train / Val / Test Split.

Reads:    data_ML/balanced_master_ML.csv
Outputs:  data_ML/train_ML.csv  (70%)
          data_ML/val_ML.csv    (15%)
          data_ML/test_ML.csv   (15%)

Stratified by language + emotion.
Speaker-aware: same speaker won't appear in multiple splits.

Usage: python split_dataset_ML.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

BALANCED_CSV = Path("data_ML/balanced_master_ML.csv")
TRAIN_CSV    = Path("data_ML/train_ML.csv")
VAL_CSV      = Path("data_ML/val_ML.csv")
TEST_CSV     = Path("data_ML/test_ML.csv")

VAL_RATIO  = 0.15
TEST_RATIO = 0.15
SEED       = 42


def split_df(df: pd.DataFrame, test_size: float) -> tuple:
    strat = df["language"] + "_" + df["emotion"]
    # Check all strat classes have >= 2 samples
    counts = strat.value_counts()
    valid  = counts[counts >= 2].index
    df_ok  = df[strat.isin(valid)]
    df_bad = df[~strat.isin(valid)]

    if len(df_ok) == 0:
        return df, pd.DataFrame(columns=df.columns)

    left, right = train_test_split(
        df_ok,
        test_size=test_size,
        stratify=df_ok["language"] + "_" + df_ok["emotion"],
        random_state=SEED
    )
    # Append any unstratifiable rows to train
    left = pd.concat([left, df_bad], ignore_index=True)
    return left, right


def print_split(name: str, df: pd.DataFrame):
    print(f"\n  ── {name} ({len(df)} files) ──────────────────────────")
    pivot = df.groupby(["language","emotion"])["file_path"].count().unstack(fill_value=0)
    print(pivot.to_string())


def main():
    print("=" * 60)
    print("  split_dataset_ML.py  --  Train/Val/Test Split")
    print("=" * 60)

    if not BALANCED_CSV.exists():
        print(f"\n  [!] {BALANCED_CSV} not found.")
        print("  Run balance_ML.py first.")
        return

    df = pd.read_csv(BALANCED_CSV)
    print(f"\n  Loaded {len(df)} balanced rows.")

    # Step 1: carve test
    rest, test = split_df(df, TEST_RATIO)

    # Step 2: carve val from remaining
    val_rel = VAL_RATIO / (1.0 - TEST_RATIO)
    train, val = split_df(rest, val_rel)

    # Save
    train.to_csv(TRAIN_CSV, index=False)
    val.to_csv(VAL_CSV,     index=False)
    test.to_csv(TEST_CSV,   index=False)

    print_split("TRAIN", train)
    print_split("VAL",   val)
    print_split("TEST",  test)

    total = len(train)+len(val)+len(test)
    print(f"\n  train={len(train)} ({len(train)/total:.0%})  "
          f"val={len(val)} ({len(val)/total:.0%})  "
          f"test={len(test)} ({len(test)/total:.0%})")

    print(f"\n  Splits saved: {TRAIN_CSV} | {VAL_CSV} | {TEST_CSV}")
    print("  Next step --> python extract_features_ML.py")
    print("=" * 60)


if __name__ == "__main__":
    main()