import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def prepare_xy(df, feature_col='Feature', target_col='Target'):
    X = df[[feature_col]]
    y = df[target_col]
    return X, y

if __name__ == "__main__":
    df = load_data('test.csv')
    X, y = prepare_xy(df)
    print(df.head())
