from sklearn.linear_model import LinearRegression

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)

if __name__ == "__main__":
    from data_load import load_data, prepare_xy
    df = load_data('test.csv')
    X, y = prepare_xy(df)
    model = train_model(X, y)
    y_pred = predict(model, X)
    print(y_pred[:5])
