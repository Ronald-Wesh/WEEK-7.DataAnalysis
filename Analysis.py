# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the dataset
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows
print(df.head())

# Check data types and missing values
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# Task 2: Basic Data Analysis
print("\nDescriptive statistics:\n", df.describe())

# Grouping by species and getting mean of each feature
print("\nMean by species:\n", df.groupby("species").mean())

# Task 3: Data Visualization

# Line plot (Fake time series using index as 'time')
plt.figure(figsize=(8,5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title("Line Chart: Sepal Length over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.grid(True)
plt.show()

# Bar chart: Average petal length per species
avg_petal_length = df.groupby("species")["petal length (cm)"].mean()
avg_petal_length.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.title("Bar Chart: Avg Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# Histogram: Distribution of sepal width
plt.hist(df['sepal width (cm)'], bins=10, color='orchid', edgecolor='black')
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot: Sepal vs Petal length
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()
