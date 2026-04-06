import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_dataframe(data):
    return pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()


def split_target(df):
    target_col = df.columns[-1]
    return df.drop(columns=[target_col]), df[target_col]


def detect_feature_types(X):
    numeric, categorical, text = [], [], []
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            numeric.append(col)
        else:
            avg_len = X[col].astype(str).str.len().mean()
            text.append(col) if avg_len > 20 else categorical.append(col)
    return numeric, categorical, text


def preprocess_features(X, numeric, categorical, text):
    X[numeric] = X[numeric].fillna(0)
    X[categorical] = X[categorical].fillna("missing")
    X[text] = X[text].fillna("")

    for col in categorical:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if numeric:
        X[numeric] = StandardScaler().fit_transform(X[numeric])

    if text:
        tfidf = TfidfVectorizer(max_features=100)
        text_data = tfidf.fit_transform(X[text[0]]).toarray()
        X = X.drop(columns=text)
        X = pd.concat([X.reset_index(drop=True), pd.DataFrame(text_data)], axis=1)

    return X


def train_and_select_model(X_train, X_test, y_train, y_test):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=50)
    }

    results = []
    best_model = ""
    best_score = 0
    best_preds = None
    best_model_obj = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        score = ((acc + f1) / 2) * 100

        results.append({
            "model": name,
            "accuracy": acc,
            "f1_score": f1,
            "score": score
        })

        if score > best_score:
            best_score = score
            best_model = name
            best_preds = preds
            best_model_obj = model

    cm = confusion_matrix(y_test, best_preds)

    feature_importance = None
    if hasattr(best_model_obj, "feature_importances_"):
        feature_importance = best_model_obj.feature_importances_

    return best_model, best_score, results, cm, feature_importance


def run_pipeline(data):

    df = prepare_dataframe(data)

    if len(df) > 50000:
        df = df.sample(50000, random_state=42)

    if df.empty:
        return None, [], None, None, None, None, None

    X, y = split_target(df)
    numeric, categorical, text = detect_feature_types(X)
    X = preprocess_features(X, numeric, categorical, text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_model, best_score, results, cm, feature_importance = train_and_select_model(
        X_train, X_test, y_train, y_test
    )

    return (
        {"best_model": best_model, "score": best_score},
        results,
        cm,
        df.head(),
        df.isnull().sum(),
        df.corr(numeric_only=True),
        feature_importance
    )