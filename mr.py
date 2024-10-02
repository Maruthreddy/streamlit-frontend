import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Function to load the dataset from a file path
def load_data():
    # Modify this path to match the location of your CSV file
    return pd.read_csv(r'C:\Users\admin\Downloads\emp_sal.csv')

# Function to perform linear regression
def perform_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to perform polynomial regression
def perform_polynomial_regression(X, y, degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model, poly_features

# Function to make predictions
def make_prediction(model, X):
    return model.predict(X)

# Main function for Streamlit interface
def main():
    st.title('Salary Prediction (Linear vs Polynomial Regression)')
    
    # Automatically load the dataset
    dataset = load_data()
    st.write("Dataset Preview:")
    st.dataframe(dataset)  # Display the dataset directly

    # Extract features and labels
    X = dataset.iloc[:, 1:2].values  # Independent variable (Position level)
    y = dataset.iloc[:, 2].values    # Dependent variable (Salary)
    
    # Linear Regression
    lin_reg_model = perform_linear_regression(X, y)
    
    # Polynomial Regression
    degree = st.slider('Select Polynomial Degree:', min_value=1, max_value=10, value=6)
    poly_reg_model, poly_features = perform_polynomial_regression(X, y, degree)
    
    # Plot Linear Regression
    st.subheader("Linear Regression")
    fig1, ax1 = plt.subplots()
    ax1.scatter(X, y, color='red')
    ax1.plot(X, make_prediction(lin_reg_model, X), color='blue')
    ax1.set_title('Linear Regression')
    ax1.set_xlabel('Position Level')
    ax1.set_ylabel('Salary')
    st.pyplot(fig1)
    
    # Plot Polynomial Regression
    st.subheader("Polynomial Regression")
    fig2, ax2 = plt.subplots()
    ax2.scatter(X, y, color='red')
    ax2.plot(X, make_prediction(poly_reg_model, poly_features.fit_transform(X)), color='blue')
    ax2.set_title(f'Polynomial Regression (Degree {degree})')
    ax2.set_xlabel('Position Level')
    ax2.set_ylabel('Salary')
    st.pyplot(fig2)
    
    # Predict salary for a specific position level
    position_level = st.number_input('Enter Position Level for Salary Prediction:', min_value=1.0, max_value=10.0, value=6.5)
    
    # Linear model prediction
    lin_pred = make_prediction(lin_reg_model, [[position_level]])
    st.write(f"Linear Regression Prediction for level {position_level}: {lin_pred[0]}")
    
    # Polynomial model prediction
    poly_pred = make_prediction(poly_reg_model, poly_features.fit_transform([[position_level]]))
    st.write(f"Polynomial Regression Prediction for level {position_level}: {poly_pred[0]}")

# Run the app
if __name__ == '__main__':
    main()
