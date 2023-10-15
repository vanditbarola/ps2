import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("data.csv")

# Convert the 'date' column to a datetime object
df['date'] = pd.to_datetime(df['date'])

# Set 'date' as the index
df.set_index('date', inplace=True)

# Define different disease types
disease_types = ["Dengue", "Malaria", "Cholera", "Flu"]

for disease_type in disease_types:
    disease_data = df[df['disease_type'] == disease_type]

    # Step 1: Descriptive Statistics
    mean_cases = disease_data['disease_cases'].mean()
    std_dev_cases = disease_data['disease_cases'].std()

    print(f"Mean of {disease_type} Cases: {mean_cases:.2f}")
    print(f"Standard Deviation of {disease_type} Cases: {std_dev_cases:.2f}")

    # Step 2: Probability Modeling
    lambda_poisson = mean_cases
    poisson_dist = poisson(mu=lambda_poisson)

    # Step 3: Probability Density Function (PDF)
    x = np.arange(0, disease_data['disease_cases'].max() + 1)
    pdf = poisson_dist.pmf(x)

    # Step 4: Cumulative Distribution Function (CDF)
    cdf = poisson_dist.cdf(x)

    # Step 5: Visualize PDF and CDF
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(x, pdf, width=0.6, align='center', alpha=0.7, label='PDF', color='skyblue')
    plt.title(f'{disease_type} - Probability Density Function (PDF)')
    plt.xlabel('Disease Cases')
    plt.ylabel('Probability')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, cdf, marker='o', linestyle='-', color='salmon')
    plt.title(f'{disease_type} - Cumulative Distribution Function (CDF)')
    plt.xlabel('Disease Cases')
    plt.ylabel('Cumulative Probability')

    plt.tight_layout()
    plt.show()

    # Step 6: Temporal Analysis
    result = sm.tsa.seasonal_decompose(disease_data['disease_cases'], model='additive', period=365)  # Assuming yearly seasonality

    # Improve visualization of the decomposition
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    result.observed.plot(ax=axes[0])
    axes[0].set_ylabel('Observed')
    result.trend.plot(ax=axes[1])
    axes[1].set_ylabel('Trend')
    result.seasonal.plot(ax=axes[2])
    axes[2].set_ylabel('Seasonal')
    result.resid.plot(ax=axes[3])
    axes[3].set_ylabel('Residual')

    plt.xlabel('Date')
    plt.title(f'{disease_type} - Seasonal Decomposition')
    plt.tight_layout()
    plt.show()
