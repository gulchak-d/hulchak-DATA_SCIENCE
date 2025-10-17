import matplotlib.pyplot as plt

def plot_regression(X, y, y_pred, title='Лінійна регресія'):
    plt.scatter(X, y, color='blue', label='Фактичні дані')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Лінія регресії')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from data_load import load_data, prepare_xy
    from regression_model import train_model, predict
    df = load_data('test.csv')
    X, y = prepare_xy(df)
    model = train_model(X, y)
    y_pred = predict(model, X)
    plot_regression(X, y, y_pred)
