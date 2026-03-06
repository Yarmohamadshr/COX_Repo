import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns

def plot_confusion_matrix(model, X_test, y_test):

    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()

    plt.savefig("outputs/confusion_matrix.png")

def plot_feature_importance(model, X):

    importance = model.feature_importances_

    features = pd.Series(importance, index=X.columns)

    features = features.sort_values(ascending=False)

    sns.barplot(x=features.values, y=features.index)

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig("outputs/feature_importance.png")