def predict_path(model, df, features, target):
    X = df[features].copy()
    y = model.predict(X)

    X[target] = y
    return X
