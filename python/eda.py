import pandas as pd

# Load dataset
df = pd.read_csv('../data/titanic.csv')

# Basic view
print("\nFirst 5 rows:\n", df.head())
print("\nDataset Shape:", df.shape)
print("\nColumn Information:\n")
print(df.info())

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert Sex to numeric (0=male, 1=female)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Print after cleaning
print("\nMissing values after cleaning:\n", df.isnull().sum())
print("\nUnique values in Embarked:", df['Embarked'].unique())

# Survival Rate overall
survival_rate = df['Survived'].mean() * 100
print(f"\nOverall Survival Rate: {survival_rate:.2f}%")

# Survival by Gender
gender_survival = df.groupby('Sex')['Survived'].mean() * 100
print("\nSurvival Rate by Gender:\n", gender_survival)

# Survival by Passenger Class
class_survival = df.groupby('Pclass')['Survived'].mean() * 100
print("\nSurvival Rate by Class:\n", class_survival)

import matplotlib.pyplot as plt

# Plot survival by gender
gender_survival.plot(kind='bar', color=['blue', 'pink'])
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Rate (%)")
plt.xticks([0, 1], ["Male", "Female"])
plt.tight_layout()
plt.savefig('../images/survival_by_gender.png')
plt.close()

# Plot survival by class
class_survival.plot(kind='bar')
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.xlabel("Class")
plt.tight_layout()
plt.savefig('../images/survival_by_class.png')
plt.close()

print("\nCharts saved in images folder.")
