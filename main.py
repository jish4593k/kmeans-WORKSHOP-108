import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class DataMiningApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Data Mining Toolkit")

        self.load_data_button = tk.Button(master, text="Load Data", command=self.load_data)
        self.load_data_button.pack()

        self.linear_regression_button = tk.Button(master, text="Perform Linear Regression", command=self.perform_linear_regression)
        self.linear_regression_button.pack()

        self.train_nn_button = tk.Button(master, text="Train Neural Network", command=self.train_neural_network)
        self.train_nn_button.pack()

    def load_data(self):
        file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            tk.messagebox.showinfo("Success", "Data loaded successfully!")

    def perform_linear_regression(self):
        if hasattr(self, 'df'):
            x_col = 'X'  # Adjust this based on your dataset
            y_col = 'Y'  # Adjust this based on your dataset

            X = self.df[[x_col]]
            y = self.df[y_col]

            # Linear Regression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            # Plotting
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=x_col, y=y_col, data=self.df, label='Actual Data')
            sns.lineplot(x=X[x_col], y=y_pred, color='red', label='Linear Regression')
            plt.title('Linear Regression')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.legend()
            plt.show()
        else:
            tk.messagebox.showerror("Error", "Load data first!")

    def train_neural_network(self):
        if hasattr(self, 'df'):
            x_col = 'X'  # Adjust this based on your dataset
            y_col = 'Y'  # Adjust this based on your dataset

            X = self.df[[x_col]].values
            y = self.df[y_col].values

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Convert data to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

            # Define a simple neural network model
            class LinearRegressionModel(nn.Module):
                def __init__(self, input_size, output_size):
                    super(LinearRegressionModel, self).__init__()
                    self.linear = nn.Linear(input_size, output_size)

                def forward(self, x):
                    return self.linear(x)

            input_size = 1  # Adjust this based on the number of features in your dataset
            output_size = 1  # Adjust this based on the number of output values

            model = LinearRegressionModel(input_size, output_size)

            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            # Training the model
            num_epochs = 1000
            for epoch in range(num_epochs):
                # Forward pass
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Make predictions on test data
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_pred_tensor = model(X_test_tensor).detach().numpy()

            # Plotting
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_test, mode='markers', name='Actual Data'))
            fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_pred_tensor.flatten(), mode='lines', name='Neural Network Prediction'))

            fig.update_layout(title='Neural Network Regression',
                              xaxis_title=x_col,
                              yaxis_title=y_col)
            fig.show()

        else:
            tk.messagebox.showerror("Error", "Load data first!")


if __name__ == "__main__":
    root = tk.Tk()
    app = DataMiningApp(root)
    root.mainloop()
