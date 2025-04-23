from main import df, np, pd, plt, sns, PCA, StandardScaler

# a) Display the first and last 12 rows of the dataset.
print(df.head(12))
print(df.tail(12))

# b) Identify and print the total number of rows and columns present.
print(f"Rows and columns: {df.shape}")

# c) List all column names along with their corresponding data types.
print(df.dtypes)

# d) Print the name of the first column.
print(f"First column: {df.columns[0]}")

# e) Generate a summary of the dataset, including non-null counts and data types.
print(df.info())

# f) Choose a categorical attribute and display the distinct values it contains.
print(df['Job_Role'].unique())  # Ensure 'Category' is an actual column in your dataset

# g) Identify the most frequently occurring value in the chosen categorical attribute.
print(df['Job_Role'].mode()[0])

# h) Calculate and present the mean, median, standard deviation, and percentiles.
for column in df.columns:
    if np.issubdtype(df[column].dtype, np.number):
        print(f"\nStatistics for '{column}':")

        print(f"Mean: {df[column].mean()}")
        print(f"Median: {df[column].median()}")
        print(f"Standard Deviation: {df[column].std()}")
        print(f"Percentiles (20%):\n{df[column].quantile(0.20)}")

# a) Apply a filter to select rows based on a specific condition of your choice (e.g., select records where a value exceeds a certain threshold).
filtered_df = df[df['Monthly_Income'] > 15000]
print(filtered_df)

# b) Identify records where a chosen attribute starts with a specific letter and count how many records match this condition.
starts_with_a = df[df['Department'].str.startswith('S')]
print(starts_with_a.shape[0])

# c) Determine the total number of duplicate rows and remove them if found
duplicate_count = df.duplicated().sum()
print(duplicate_count)

# d) Convert the data type of a numerical column from integer to string.
df['Project_Count'] = df['Project_Count'].astype(str)
print(df.dtypes)


grouped = df.groupby(['Department','Job_Role'])

# f) Check for the existence of missing values within the dataset.
missing_values = df.isnull().sum()
print(missing_values)


bins = pd.cut(df['Job_Level'].astype(float), bins=5)
bin_counts = bins.value_counts()
print(bin_counts)

# i) Identify and print the row corresponding to the maximum value of a selected numerical feature.
print(df.loc[df['Monthly_Income'].idxmax()])
print('Income: ')
print(df.Monthly_Income.max())

#j) Construct a boxplot for an attribute you consider significant and justify the selection.
#because it is one of, if not the main factor when it comes to switching jobs
sns.boxplot(x =df['Monthly_Income'])
sns.set(style="whitegrid")
plt.title('boxplot of Monthly Income')
plt.show()

#k) Generate a histogram for a chosen attribute and provide an explanation for its relevance.
#it is significant because the amount of training shows the strength of a company
print(df.columns)
df['Training_Hours_Last_Year'] = df['Training_Hours_Last_Year'].astype(float)
plt.hist(df['Training_Hours_Last_Year'], bins=200, edgecolor='black')
plt.title('Histogram of Monthly Income')
plt.xlabel('income')
plt.ylabel('Frequency')
plt.show()

df_clean = df[(df['Hourly_Rate'] >= 20) &
               (df['Hourly_Rate'] <= 40) &
               (df['Years_at_Company'] <= 10)]

sns.lmplot(x='Years_at_Company', y='Hourly_Rate', data=df_clean, height=4, aspect=1.5, scatter_kws={'alpha': 0.2, 's': 10})

plt.title('Scatterplot with Regression Line: Hourly Rate vs Years at a Company')
plt.ylabel('Hourly Rate')
plt.xlabel('Years at Company')
plt.show()

# Assuming df is your DataFrame
# Select numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numerical_columns]

# Normalize the data using StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Scatterplot before PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_numeric['Average_Hours_Worked_Per_Week'], y=df_numeric['Hourly_Rate'])
plt.xlabel("Average_Hours_Worked_Per_Week")
plt.ylabel("Hourly_Rate")
plt.title("Scatterplot of Average_Hours_Worked_Per_Week vs. Hourly_Rate Before PCA")
plt.show()

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Convert to DataFrame for visualization
selected_features = df_numeric[['Average_Hours_Worked_Per_Week', 'Hourly_Rate']]
selected_scaled = scaler.fit_transform(selected_features)

pca = PCA(n_components=2)
df_pca_selected = pca.fit_transform(selected_scaled)

df_pca_selected = pd.DataFrame(df_pca_selected, columns=['PC1', 'PC2'])

# Scatterplot after PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca_selected['PC1'], y=df_pca_selected['PC2'])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Scatterplot After PCA")
plt.show()

# Plot principal component axes (eigenvectors)
plt.figure(figsize=(8, 6))

origin = np.array([0, 0])  # Origin for vectors
eigenvectors = pca.components_  # Principal component directions

# Plot each principal component
for i in range(2):
    plt.arrow(origin[0], origin[1], eigenvectors[i, 0], eigenvectors[i, 1],
              head_width=0.1, head_length=0.1, color=['r', 'b'][i], label=f"PC{i+1}")

plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Principal Component Axes (Eigenvectors)")
plt.grid()
plt.show()

# o) Analyze the correlation between numerical features using a heatmap.

print(df.columns)

# Corrected: Removed the unpacking operator (*) which was causing a syntax error
numerical_features = [
    'Employee_ID',
    'Job_Level', 'Monthly_Income', 'Hourly_Rate',
    'Years_at_Company', 'Years_in_Current_Role',
    'Years_Since_Last_Promotion',
    'Performance_Rating', 'Training_Hours_Last_Year',
    'Average_Hours_Worked_Per_Week', 'Absenteeism',
    'Work_Environment_Satisfaction', 'Relationship_with_Manager',
    'Job_Involvement', 'Distance_From_Home', 'Number_of_Companies_Worked',
]


# Convert 'Project_Count' back to numeric if needed for correlation
df['Project_Count'] = pd.to_numeric(df['Project_Count'], errors='coerce')
# errors='coerce' will handle any non-numeric values by setting them to NaN
numerical_features.append('Project_Count')  # Include Project_Count in the features for correlation

correlation_matrix = df[numerical_features].corr()

plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# a) Use Python to calculate and display the correlation matrix, and identify potential features relevant for classification

correlation_matrix = df[numerical_features].corr()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# b) Use Python to find the class distribution of a selected categorical feature and analyze the results.

sns.countplot(x='Department', data=df)
plt.xticks(rotation=45)
plt.xlabel('Department')
plt.title('Department Distribution')
plt.show()

print()
print('so, this shows that the marketting team had the most people whereas the IT team had the least of them...')
print('in all cases, none of them have less than 1750 employees')

# c) Apply Python techniques to create new features from existing ones (feature engineering) and explain the significance of the new features.

#print(df.columns)

df['Experience_Salary'] = df['Monthly_Income'] / df['Number_of_Companies_Worked']

df['Training_Hours_Last_Year'] = df['Training_Hours_Last_Year'].astype(float)
plt.hist(df['Experience_Salary'], bins=200, edgecolor='black')
plt.title('Experience Salary')
plt.xlabel('Experience Salary ')
plt.ylabel('Frequency')
plt.show()

print()
print('Usually a good company pays better the more experience you have, so I wanted to see how this would look like')
print('it is very possible that if the experience does not mean a better salary that this owuld lead to a lot more dropouts')