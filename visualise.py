import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results(df, y_test, y_pred):
    # Histogram of land prices by location
    sns.histplot(
        data=df,
        x="price_per_cent",
        hue="taluk",
        bins=50,
        kde=True
    )

    plt.xlabel("Price Per Cent")
    plt.ylabel("Frequency")
    plt.title("Land Price Distribution by Location")

    plt.show()

    # Scatter plot for Actual vs Predicted prices
    sns.scatterplot(x=y_test, y=y_pred)

    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Land Prices")

    plt.show()

def plot_feature_importance(model, X_train):

    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    )

    importance = importance.sort_values(ascending=False).head(10)

    importance.plot(kind='bar')

    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Top 10 Features Affecting Land Price")

    plt.show()