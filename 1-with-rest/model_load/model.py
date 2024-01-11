from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import mlflow.sklearn
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://mlflow:5000")

with mlflow.start_run() as run:
    X, y = make_moons(noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    signature = infer_signature(X_test, y_pred)

    mlflow.log_metrics({"accuracy": accuracy_score(y_test, y_pred)})
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="sk-learn-decision-tree-reg-model",
    )
    print(run.info.run_id)