import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier, vectorizer):
        self.base_classifier = base_classifier
        self.vectorizer = vectorizer
        self.classifiers_ = {}
        self.label_encoders_ = {}

    def fit(self, X, y):
        X_transformed = self.vectorizer.fit_transform(X)
        for level in range(y.shape[1]):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y[:, level])
            self.label_encoders_[level] = le
            classifier = clone(self.base_classifier)
            classifier.fit(X_transformed, y_encoded)
            self.classifiers_[level] = classifier
        return self

    def predict(self, X):
        X_transformed = self.vectorizer.transform(X)
        predictions = np.zeros((X.shape[0], len(self.classifiers_)), dtype=object)
        for level, classifier in self.classifiers_.items():
            y_pred = classifier.predict(X_transformed)
            predictions[:, level] = self.label_encoders_[level].inverse_transform(y_pred)
        return predictions


# Load Data
train_data = pd.read_csv('./train_40k.csv', engine='pyarrow')
val_data = pd.read_csv('./val_10k.csv', engine='pyarrow')

# Prepare data for hierarchical classification
X_train = train_data['Text'].values
y_train_hierarchical = train_data[['Cat1', 'Cat2', 'Cat3']].values

X_val = val_data['Text'].values
y_val_hierarchical = val_data[['Cat1', 'Cat2', 'Cat3']].values

# Prepare data for flat classification
train_data['FlatCategory'] = train_data['Cat1'] + '/' + train_data['Cat2'] + '/' + train_data['Cat3']
y_train_flat = train_data['FlatCategory'].values

val_data['FlatCategory'] = val_data['Cat1'] + '/' + val_data['Cat2'] + '/' + val_data['Cat3']
y_val_flat = val_data['FlatCategory'].values

vectorizer = make_pipeline(TfidfVectorizer())
param_grid = {'C': [0.01, 0.1, 1, 10]}
hierarchical_clf = HierarchicalClassifier(GridSearchCV(LinearSVC(), param_grid, cv=5), vectorizer)
flat_clf = make_pipeline(TfidfVectorizer(), GridSearchCV(LinearSVC(), param_grid, cv=5))

# Train classifiers
print("Training hierarchical classifier...")
hierarchical_clf.fit(X_train, y_train_hierarchical)

print("Training flat classifier...")
flat_clf.fit(X_train, y_train_flat)

y_pred_hierarchical = hierarchical_clf.predict(X_val)
y_pred_flat = flat_clf.predict(X_val)

# Metrics
hierarchical_metrics = {metric: [] for metric in ["accuracy", "precision", "recall", "f1"]}
for i in range(y_val_hierarchical.shape[1]):
    hierarchical_metrics["accuracy"].append(accuracy_score(y_val_hierarchical[:, i], y_pred_hierarchical[:, i]))
    hierarchical_metrics["precision"].append(
        precision_score(y_val_hierarchical[:, i], y_pred_hierarchical[:, i], average='macro'))
    hierarchical_metrics["recall"].append(
        recall_score(y_val_hierarchical[:, i], y_pred_hierarchical[:, i], average='macro'))
    hierarchical_metrics["f1"].append(f1_score(y_val_hierarchical[:, i], y_pred_hierarchical[:, i], average='macro'))

hierarchical_results = {metric: np.mean(scores) for metric, scores in hierarchical_metrics.items()}

flat_results = {
    "accuracy": accuracy_score(y_val_flat, y_pred_flat),
    "precision": precision_score(y_val_flat, y_pred_flat, average='macro'),
    "recall": recall_score(y_val_flat, y_pred_flat, average='macro'),
    "f1": f1_score(y_val_flat, y_pred_flat, average='macro')
}

print("Hierarchical Classifier Results:", hierarchical_results)
print("Flat Classifier Results:", flat_results)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class Review(BaseModel):
    text: str


@app.post("/predict/")
def predict(review: Review):
    prediction = hierarchical_clf.predict([review.text])
    return {"prediction": prediction.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", reload=True, port=8000, log_level="info")
