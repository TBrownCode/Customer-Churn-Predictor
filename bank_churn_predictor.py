import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

#load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    print("Dataset shape:", df.shape)
    print("\nColumn names:", df.columns)
    print("\nUnique values in Geography:", df['Geography'].unique())
    
    #drop unnecessary columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    #convert categorical variables to numeric
    geography_dummies = pd.get_dummies(df['Geography'], prefix='Geography', drop_first=True)
    df = pd.concat([df, geography_dummies], axis=1)
    df = df.drop('Geography', axis=1)
    
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    
    #split features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    #split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #get feature names before scaling
    feature_names = X.columns.tolist()
    
    #scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df, feature_names

#train logistic regression model
def train_model(X_train, y_train):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    model = LogisticRegression(random_state=42, C=1.0, class_weight=class_weight_dict)
    model.fit(X_train, y_train)
    return model

#evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred

#create visualizations
def create_visualizations(df, model, X_test, y_test, y_pred):
    #visualization 1: correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap of Features")
    plt.savefig('correlation_heatmap.png')
    plt.close()

    #visualization 2: feature importance
    feature_importance = pd.DataFrame({'feature': df.drop('Exited', axis=1).columns, 'importance': abs(model.coef_[0])})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title("Feature Importance")
    plt.savefig('feature_importance.png')
    plt.close()

    #visualization 3: confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig('confusion_matrix.png')
    plt.close()

#user interface
def user_interface(model, scaler, feature_names):
    print("\nCustomer Churn Prediction Tool")
    print("Please enter the following customer information:")
    
    credit_score = float(input("Credit Score (300-850): "))
    geography = input("Geography (Germany/France/Spain): ")
    gender = input("Gender (Female/Male): ")
    age = float(input("Age: "))
    tenure = float(input("Tenure (years with the bank): "))
    balance = float(input("Balance: (format 10000.00 for 10,000) "))
    num_of_products = float(input("Number of Products: "))
    has_credit_card = float(input("Has Credit Card (0 for no / 1 for yes):  "))
    is_active_member = float(input("Is Active Member (0 for no / 1 for yes): "))
    estimated_salary = float(input("Estimated Salary: (format 30000.00 for 30,000) "))
    
    #prepare input data
    input_data = {
        'CreditScore': credit_score,
        'Gender': 1 if gender.lower() == 'male' else 0,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_credit_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
        'Geography_Germany': 1 if geography.lower() == 'germany' else 0,
        'Geography_Spain': 1 if geography.lower() == 'spain' else 0
    }
    
    input_df = pd.DataFrame([input_data])[feature_names]

    input_data_scaled = scaler.transform(input_df)
    
    #make prediction
    raw_prediction = model.decision_function(input_data_scaled)[0]
    prediction = model.predict(input_data_scaled)[0]
    probability = 1 / (1 + np.exp(-raw_prediction))
    
    print(f"\nRaw prediction score: {raw_prediction:.4f}")
    print(f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}")
    print(f"Probability of Churn: {probability:.4f}")

    #print feature importances
    coefficients = model.coef_[0]
    for name, coef in zip(feature_names, coefficients):
        print(f"{name}: {coef:.4f}")

#main function
def main():
    file_path = '1000 rows.csv'
    X_train, X_test, y_train, y_test, scaler, df, feature_names = load_and_preprocess_data(file_path)
    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    create_visualizations(df, model, X_test, y_test, y_pred)

    print("Feature names:", feature_names)
    print("Class distribution in training data:")
    print(y_train.value_counts(normalize=True))
    
    while True:
        user_interface(model, scaler, feature_names)
        if input("\nDo you want to make another prediction? (yes/no): ").lower() != 'yes':
            break

if __name__ == "__main__":
    main()

