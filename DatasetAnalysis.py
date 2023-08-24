import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)

# Load data
df = pd.read_csv('data.csv')

# Display the first few rows of the dataframe
print(df.head())

# Basic aggregate statistics
print(df.describe())

# Checking for missing values
print(df.isnull().sum())

# Correlations between different columns
correlations = df.corr()
print(correlations)

# Visualizing the correlations
plt.figure(figsize=(10,6))
sns.heatmap(correlations, annot=True, cmap="coolwarm")
plt.show()

# Group data by country and calculate the total number of deaths and cases for each
total_deaths_by_country = df.groupby('Entity')['Deaths'].sum().sort_values(ascending=False)
total_cases_by_country = df.groupby('Entity')['Cases'].sum().sort_values(ascending=False)

# Get countries sorted by total deaths and cases
countries_sorted_by_deaths = total_deaths_by_country.index
countries_sorted_by_cases = total_cases_by_country.index

# Visualize deaths over time per country
plt.figure(figsize=(10,6))
for country in countries_sorted_by_deaths:
    country_data = df[df['Entity'] == country]
    plt.plot(country_data['Date'], country_data['Deaths'], label=country)
plt.legend()
plt.show()

# Visualize cases over time per country
plt.figure(figsize=(10,6))
for country in countries_sorted_by_cases:
    country_data = df[df['Entity'] == country]
    plt.plot(country_data['Date'], country_data['Cases'], label=country)
plt.legend()
plt.show()

# Histogram of GDP per Capita
plt.figure(figsize=(10,6))
plt.hist(df['GDP/Capita'], bins=30)
plt.xlabel('GDP/Capita')
plt.ylabel('Frequency')
plt.title('Histogram of GDP/Capita')
plt.show()

# Scatter plot of 'Cases' vs 'Deaths'
plt.figure(figsize=(10,6))
plt.scatter(df['Cases'], df['Deaths'])
plt.xlabel('Cases')
plt.ylabel('Deaths')
plt.title('Scatter plot of Cases vs Deaths')
plt.show()


# Create a copy of df with renamed columns
df_renamed = df.rename(columns={
    'Hospital beds per 1000 people': 'Hospitals',
    'Medical doctors per 1000 people': 'Doctors',
})

# Columns for pairplot
selected_columns_1 = ['Cases', 'Deaths', 'Daily tests', 'GDP/Capita']
selected_columns_2 = ['Cases', 'Deaths', 'Doctors', 'Hospitals']

# Pairplot
sns.pairplot(df_renamed[selected_columns_1])
plt.show()

sns.pairplot(df_renamed[selected_columns_2])
plt.show()

