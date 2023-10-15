import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv("data.csv")  # Replace with your actual data file

# Convert the 'date' column to a datetime object
df['date'] = pd.to_datetime(df['date'])

# Calculate the total number of months in your dataset
total_months = (df['date'].max().year - df['date'].min().year) * 12 + df['date'].max().month - df['date'].min().month + 1

# List of disease types
disease_types = df['disease_type'].unique()

# Create a plot for each disease type
plt.figure(figsize=(12, 6))

for disease_type in disease_types:
    # Filter data for the specific disease
    disease_data = df[df['disease_type'] == disease_type]

    # Group data by month and calculate the probability of disease occurrence in each month
    monthly_probabilities = disease_data.groupby(disease_data['date'].dt.month)['disease_cases'].sum() / total_months

    # Plot the probabilities for this disease
    plt.plot(monthly_probabilities.index, monthly_probabilities, label=disease_type)

# Customize the plot
plt.xlabel('Month')
plt.ylabel('Probability of Disease Occurrence %')
plt.title('Monthly Probability of Disease Occurrence in')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.show()