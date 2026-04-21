import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# -----------------------------
# 1. Load Dataset
# -----------------------------
def load_data():
    data = {
        "Attendance_Percentage": [85, 78, 92, 70, 88, 55, 40],
        "Mess_Usage_Percentage": [80, 75, 90, 65, 85, 50, 35],
        "Meals_Consumed": [400, 370, 455, 340, 430, 260, 190]
    }
    df = pd.DataFrame(data)
    return df


# -----------------------------
# 2. Train Model
# -----------------------------
def train_model():
    df = load_data()

    X = df[["Attendance_Percentage"]]
    y = df["Meals_Consumed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model (optional but good for hackathon)
    joblib.dump(model, "mess_model.pkl")

    return model


# -----------------------------
# 3. Predict Function
# -----------------------------
def predict_meals(attendance_percentage):
    try:
        model = joblib.load("mess_model.pkl")
    except:
        model = train_model()

    prediction = model.predict([[attendance_percentage]])
    return int(prediction[0])


# -----------------------------
# 4. Example Run
# -----------------------------
if __name__ == "__main__":
    model = train_model()

    print("📊 Model trained successfully!")

    test_attendance = 82
    predicted_meals = predict_meals(test_attendance)

    print(f"🔮 Predicted Meals for {test_attendance}% attendance: {predicted_meals}")
