import pandas as pd
import numpy as np


def american_to_probability(odds: float) -> float:
    if odds is None or pd.isna(odds) or odds == 0 : 
        return float('nan')
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)
""" Convert the American moneyline odds to implied probabilities taking into account possible ZeroDivisionError """

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NBA betting dataset:
    - Create outcome variable (1 = win, 0 = loss)
    - Convert moneyline odds to probabilities
    - Drop rows with missing or invalid data
    """

    df = df.copy()

    # Create binary target (win)
    df["win"] = (df["score"] > df["opponentScore"]).astype(int)

    # Convert moneyline odds to probabilities
    df["team_prob"] = df["moneyLine"].apply(american_to_probability)
    df["opp_prob"] = df["opponentMoneyLine"].apply(american_to_probability)

    # Remove rows with missing probabilities (if any)
    df = df.dropna(subset=["team_prob", "opp_prob", "win"])

    return df


if __name__ == "__main__":
    from data_loading import load_raw_data

    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)
# basic info about the cleaned dataset
    print("Raw shape:", df_raw.shape)
    print("Clean shape:", df_clean.shape)
    print(df_clean.head())

    df_clean.to_csv("data/cleaned_data.csv", index = False) #save the clean dataset
