from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate
from src.visualization import plot_confusion_matrix, plot_feature_importance


data_path = "data/telecom_churn_sample.csv"

df = load_data(data_path)
X, y = preprocess_data(df)

model, X_test, y_test = train_model(X, y)

acc, auc = evaluate(model, X_test, y_test)

print("Accuracy:", acc)
print("ROC AUC:", auc)
plot_confusion_matrix(model, X_test, y_test)
plot_feature_importance(model, X)
