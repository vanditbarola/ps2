import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate random data for a dataset spanning 10 years
n = 10 * 365  # Number of data points for 10 years
locations = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Gurgaon", "Faridabad", "Ghaziabad", "Noida"]
# Simulate date values spanning 10 years with a fixed frequency of 1 day
start_date = "2010-01-01"
end_date = "2019-12-31"
date_range = pd.date_range(start=start_date, end=end_date, periods=n)

# Define different disease types
disease_types = ["Dengue", "Malaria", "Cholera", "Flu"]

# Create an empty list to store the disease type, location, and other variables for each data point
disease_type_list = []
location_list = []

# Generate disease cases for each disease type and assign a disease type to each data point with higher variability
for _ in range(n):
    disease_type = np.random.choice(disease_types)
    disease_type_list.append(disease_type)
    
    location = np.random.choice(locations)  # Assign a random location from the list
    location_list.append(location)

# Simulate disease cases with even higher variability
disease_cases = np.random.randint(0, 150, size=n)

# Simulate more variable weather data with larger ranges
temperature = np.random.uniform(0, 50, size=n)
humidity = np.random.uniform(10, 90, size=n)
rainfall = np.random.uniform(0, 50, size=n)

# Simulate population data with larger variability
population = np.random.randint(100000, 3000000, size=n)

# Create a DataFrame
data = {
    "date": date_range,
    "location": location_list,
    "disease_type": disease_type_list,
    "disease_cases": disease_cases,
    "temperature": temperature,
    "humidity": humidity,
    "rainfall": rainfall,
    "population": population,
}

df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv("data.csv", index=False)