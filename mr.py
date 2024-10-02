import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def load_data():
    url = 'https://raw.githubusercontent.com/Maruthreddy/streamlit-frontend/main/emp_sal.csv'
    return pd.read_csv(url)

def perform_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def perform_polynomial_regression(X, y, degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model, poly_features

def make_prediction(model, X):
    return model.predict(X)

def main():
    st.title('Salary Prediction (Linear vs Polynomial Regression)')
    dataset = load_data()
    st.write("Dataset Preview:")
    st.dataframe(dataset)

    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values

    lin_reg_model = perform_linear_regression(X, y)
    degree = st.slider('Select Polynomial Degree:', min_value=1, max_value=10, value=6)
    poly_reg_model, poly_features = perform_polynomial_regression(X, y, degree)

    st.subheader("Linear Regression")
    fig1, ax1 = plt.subplots()
    ax1.scatter(X, y, color='red')
    ax1.plot(X, make_prediction(lin_reg_model, X), color='blue')
    ax1.set_title('Linear Regression')
    ax1.set_xlabel('Position Level')
    ax1.set_ylabel('Salary')
    st.pyplot(fig1)

    st.subheader("Polynomial Regression")
    fig2, ax2 = plt.subplots()
    ax2.scatter(X, y, color='red')
    ax2.plot(X, make_prediction(poly_reg_model, poly_features.fit_transform(X)), color='blue')
    ax2.set_title(f'Polynomial Regression (Degree {degree})')
    ax2.set_xlabel('Position Level')
    ax2.set_ylabel('Salary')
    st.pyplot(fig2)

    position_level = st.number_input('Enter Position Level for Salary Prediction:', min_value=1.0, max_value=10.0, value=6.5)
    lin_pred = make_prediction(lin_reg_model, [[position_level]])
    st.write(f"Linear Regression Prediction for level {position_level}: {lin_pred[0]}")
    
    poly_pred = make_prediction(poly_reg_model, poly_features.fit_transform([[position_level]]))
    st.write(f"Polynomial Regression Prediction for level {position_level}: {poly_pred[0]}")

if __name__ == '__main__':
    main()
