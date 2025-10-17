from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

if __name__ == "__main__":
    from data_load import load_data, prepare_xy
    from regression_model import train_model, predict
    df = load_data('test.csv')
    X, y = prepare_xy(df)
    model = train_model(X, y)
    y_pred = predict(model, X)
    mse, mae, r2 = evaluate(y, y_pred)
    print(mse, mae, r2)
