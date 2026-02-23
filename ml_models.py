import os
import pickle
from typing import List, Tuple, Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# directory where model artefacts will be stored
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# synthetic data generation
# These functions generate dummy data so the models have something to train on right out of the box
def generate_synthetic_emotions() -> Tuple[List[str], List[str]]:
    """Return a small synthetic dataset mapping sentences to emotion labels."""
    samples = [
        #A list of tuples, where each tuple contains a text string and its corresponding label
        ("I am so frustrated with this delay", "Frustrated"),
        ("Why are you always like this?", "Accusatory"),
        ("I guess that makes sense", "Neutral"),
        ("I'm sorry you're upset", "Empathetic"),
        ("This is unacceptable", "Frustrated"),
        ("You never listen to me", "Accusatory"),
        ("It's fine, don't worry about it", "Neutral"),
        ("I understand how that could hurt", "Empathetic"),
        ("I'm furious right now", "Frustrated"),
        ("Please stop blaming me", "Defensive"),
    ]
    texts, labels = zip(*samples)#unzip the list of tuples
    #It separates the data into two distinct lists: one list containing only the texts, and one list containing only the labels.

    return list(texts), list(labels)

def generate_synthetic_conflicts() -> Tuple[List[str], List[str]]:
    """Return a synthetic dataset for conflict "type" classification."""
    samples = [
        ("You changed the plan without telling me", "Boundary violation"),
        ("I expected the report yesterday", "Expectation mismatch"),
        ("You always do this last minute", "Boundary violation"),
        ("We had different assumptions", "Expectation mismatch"),
        ("Don't talk to me like that", "Respect issue"),
        ("You broke your promise", "Trust issue"),
        ("This isn't what I agreed to", "Expectation mismatch"),
        ("Stop touching my things", "Boundary violation"),
    ]
    texts, labels = zip(*samples)
    return list(texts), list(labels)

# model training / persistence (Dont Repeat Yourelf DRY Refactor)

def _load_or_train_model(
    #Instead of writing separate training code for emotions and conflicts
    #It takes a name, model_prefix , and one of your data generator functions
    model_prefix: str, data_generator: Callable[[], Tuple[List[str], List[str]]]
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """Generic helper to load models from disk, or train and save them if missing."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    #os.makedirs ensures the models/ folder exists. exist_ok=True prevents it from crashing if the folder is already there.
    
    vec_path = os.path.join(MODEL_DIR, f"{model_prefix}_vectorizer.pkl")
    clf_path = os.path.join(MODEL_DIR, f"{model_prefix}_classifier.pkl")
    # vec_path & clf_path defines the exact filenames where the vectorizer and classifier will be saved (e.g., emotion_vectorizer.pkl)

    # If both artifacts exist, load and return them
    if os.path.exists(vec_path) and os.path.exists(clf_path):
        with open(vec_path, "rb") as f_vec, open(clf_path, "rb") as f_clf: #rb = read binary mode
            return pickle.load(f_vec), pickle.load(f_clf)

    # Otherwise, generate data and train
    X, y = data_generator() # If the models don't exist, it runs your generator to get texts (X) and labels (y).
    vectorizer = TfidfVectorizer() 
    clf = LogisticRegression(max_iter=1000)
    
    clf.fit(vectorizer.fit_transform(X), y)

    # Save the newly trained artifacts
    with open(vec_path, "wb") as f_vec, open(clf_path, "wb") as f_clf:
        pickle.dump(vectorizer, f_vec)
        pickle.dump(clf, f_clf)
        
    return vectorizer, clf

# inference helpers

def _predict_label(text: str, vectorizer: TfidfVectorizer, clf: LogisticRegression) -> str:
    """Generic helper for prediction."""
    return clf.predict(vectorizer.transform([text]))[0]

def predict_emotion(text: str, vectorizer: TfidfVectorizer, clf: LogisticRegression) -> str:
    """Return the predicted emotion label for a single line of text."""
    return _predict_label(text, vectorizer, clf)

def predict_conflict(text: str, vectorizer: TfidfVectorizer, clf: LogisticRegression) -> str:
    """Return the predicted conflict type label for a single line of text."""
    return _predict_label(text, vectorizer, clf)

# ensure models are available when the module is imported
_emotion_vectorizer, _emotion_clf = _load_or_train_model("emotion", generate_synthetic_emotions)
_conflict_vectorizer, _conflict_clf = _load_or_train_model("conflict", generate_synthetic_conflicts)