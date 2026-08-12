"""Model registry for assignment classifiers."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def get_models(random_state: int):
    """Return configured model instances keyed by display name."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
