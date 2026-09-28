from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

from feature_extraction import load_data


def train_model():
    # Load dataset
    X, y = load_data()
    print("Loaded data shape:", X.shape)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model
    model = RandomForestClassifier()

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", accuracy)

    # Save trained model
    joblib.dump(model, "voice_model.pkl")
    print("Model saved as voice_model.pkl")


if __name__ == "__main__":
    train_model()