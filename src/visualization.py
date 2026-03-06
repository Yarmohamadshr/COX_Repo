import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns
import os

def plot_confusion_matrix(model, X_test, y_test,labels=[0,1]):

    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)

    disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

    disp.plot()

    plt.savefig("outputs/confusion_matrix.png")

def plot_feature_importance(model, X):

    os.makedirs("outputs", exist_ok=True)

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(8,5))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importance
    )

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig("outputs/feature_importance.png")