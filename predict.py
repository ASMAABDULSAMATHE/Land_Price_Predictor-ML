import pandas as pd
def evaluate(model, X_test, y_test=None):
    #Model prediction
    y_pred=model.predict(X_test)
    df_comparison=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
    print(df_comparison.head())

    return y_pred, df_comparison


