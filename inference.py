import joblib

# Load the trained model and vectorizer
model_path = "model/logistic_model.joblib"
vectorizer_path = "model/vectorizer.joblib"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Function to make predictions on user input
def predict_gender(name):
    # Convert the name to lowercase to match training preprocessing
    name = name.lower()
    
    # Transform the name using the loaded vectorizer
    name_vector = vectorizer.transform([name])
    
    # Predict the gender
    gender_pred = model.predict(name_vector)
    
    return gender_pred[0]

# Interactive inference
while True:
    user_input = input("Enter a name to predict gender (or type 'exit' to stop): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    gender = predict_gender(user_input)
    print(f"The predicted gender for '{user_input}' is: {gender.capitalize()}")
