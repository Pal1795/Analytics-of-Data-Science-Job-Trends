#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# ## Step 1: Data Loading

# In[2]:


df=pd.read_csv('Job_data_cleaned.csv') 
df.head()


# ## Step 2: Data Preprocessing

# ## Checking for Null Values

# In[3]:


print(df.isnull().sum())


# ## Handling Missing Salary Values

# In[4]:


# Remove rows where the 'salary' column has NaN values
data_df = df.dropna(subset=['salary'])


# ## Removing Hourly Salary Entries
# 

# In[5]:


# Create a copy of the DataFrame to avoid SettingWithCopyWarning
jobsdata_df = data_df.copy()

# Remove entries where the 'salary' column contains '/hr'
jobsdata_df = jobsdata_df[~jobsdata_df['salary'].str.contains('/hr')] 


# ## Identifying Duplicated Rows and Final DataFrame Length

# In[6]:


# Get the indices of duplicated rows
duplicated_indices = jobsdata_df[jobsdata_df.duplicated()].index

# Count the number of duplicated rows
num_duplicated = len(duplicated_indices)

# Get the total length of the DataFrame after checking for duplicates
final_length = len(jobsdata_df)


print("Number of duplicated rows:", num_duplicated)  
print("Final length of the DataFrame:", final_length)


# ## Extracting City and State from the Location Column

# In[7]:


jobsdata_df['City'] = jobsdata_df['location'].str.split(',').str[0].str.strip()  # Extract city
jobsdata_df['State'] = jobsdata_df['location'].str.split(',').str[-1].str.strip()  # Extract state


# In[8]:


# Display the updated DataFrame
jobsdata_df.columns


# ## Reordering Columns in the DataFrame

# In[9]:


# Reorder columns to place 'City' 'State' next to 'location'
column_order = ['job_title', 'company_name', 'location', 'City', 'State', 'time_posted', 
                'num_applicants', 'salary', 'seniority', 'employment_type', 
                'job_function', 'industry']

# Reindex the DataFrame with the new column order
jobsdata_df = jobsdata_df[column_order]

# Display the updated DataFrame
jobsdata_df.head()


# ## Standardizing State Values Based on City Names
# 

# In[10]:


for index, row in jobsdata_df.iterrows():
    if row['City'] in ['Los Angeles Metropolitan Area', 'San Francisco Bay Area', 'San Francisco', 'San Diego Metropolitan Area']:
        jobsdata_df.at[index, 'State'] = 'CA'
    elif row['City'] == 'New York':
        jobsdata_df.at[index, 'State'] = 'NY'
    elif row['City'] == 'Seattle':
        jobsdata_df.at[index, 'State'] = 'WA'
    elif row['City'] == 'Raleigh-Durham-Chapel Hill Area':
        jobsdata_df.at[index, 'State'] = 'NC'


# ## Removing Rows with Invalid 'City' and 'State' Values

# In[11]:


jobsdata_df.drop(jobsdata_df[(jobsdata_df['City'] == 'United States') & (jobsdata_df['State'] == 'United States')].index, inplace=True)


# ## Extracting Minimum and Maximum Salary from the Salary Column
# 

# In[12]:


# Extract minimum and maximum salary while ignoring decimal values
jobsdata_df[['Min_Salary', 'Max_Salary']] = jobsdata_df['salary'].str.extract(
    r'\$([\d,]+)\.\d{2}/yr - \$([\d,]+)\.\d{2}/yr'
)

# Remove commas and convert to integer
jobsdata_df['Min_Salary'] = jobsdata_df['Min_Salary'].str.replace(',', '').astype(int)
jobsdata_df['Max_Salary'] = jobsdata_df['Max_Salary'].str.replace(',', '').astype(int)

jobsdata_df.head()


# In[13]:


# Reorder columns to place 'State' next to 'location'
column_order = ['job_title', 'company_name', 'location', 'City', 'State', 'time_posted', 
                'num_applicants', 'salary', 'Min_Salary', 'Max_Salary', 'seniority', 'employment_type', 
                'job_function', 'industry']

# Reindex the DataFrame with the new column order
jobsdata_df = jobsdata_df[column_order]


# ## Industry Mapping
# 
# This DataFrame contains the mapping of different industries, including relevant details that can be utilized for analysis in the job postings dataset. 
# 
# The industry data is sourced from the [Bureau of Labor Statistics]
# 
# (https://www.bls.gov/iag/tgs/iag_index_naics.htm).
# 
# 
# 

# In[14]:


industry_df=pd.read_csv('Industry_mapping.csv') 
industry_df.head()


# ## Merging Job Postings with Industry Data
# 

# In[15]:


#Perform the join on the 'industry' column
jobsdata1_df = pd.merge(jobsdata_df, industry_df, on='industry', how='left')

# Display the merged DataFrame
jobsdata1_df.head()


# ## Job Title Mapping 
# 
# Categorizing Related Roles into Distinct Categories

# In[16]:


jobcategory_df=pd.read_csv('jobtitle_mapping.csv') # reads the test.csv file and stores its contents in the df DataFrame 
jobcategory_df.head()


# ## Merging Job Category DataFrame with Job Postings DataFrame

# In[17]:


# Joining jobcategory_df with jobsdata1_df on the 'job_title' column
job_postings_df = jobsdata1_df.merge(jobcategory_df, on='job_title', how='left')

# Display the first few rows of the merged DataFrame
job_postings_df.head()


# In[18]:


print(job_postings_df.columns)


# In[19]:


# Save the DataFrame to a CSV file
job_postings_df.to_csv('today.csv', index=False)


# ## Step 3: Data Visualization

# ## Distribution of Job Title Categories

# In[20]:


# Count the occurrences of each category in job_title_category
category_counts = job_postings_df['job_title_category'].value_counts().reset_index()
category_counts.columns = ['job_title_category', 'count']

# Create a donut chart
fig = px.pie(category_counts, 
             names='job_title_category', 
             values='count', 
             hole=0.4,  # This creates the "donut" effect
             title='Distribution of Job Title Categories',
             labels={'job_title_category': 'Job Title Category', 'count': 'Count'}
            )

# Update layout for better visualization
fig.update_traces(textinfo='percent+label')  # Show percentages and labels on the chart

# Show the plot
fig.show()


# ## Distribution of Industry Categories

# In[21]:


# Count the occurrences of each category in industry_category
industry_counts = job_postings_df['industry_category'].value_counts().reset_index()
industry_counts.columns = ['industry_category', 'count']

# Create a donut chart (pie chart with hole)
fig = px.pie(industry_counts, 
             names='industry_category', 
             values='count', 
             hole=0.4,  # This creates the "donut" effect
             title='Distribution of Industry Categories',
             labels={'industry_category': 'Industry Category', 'count': 'Count'}
            )

# Update layout for better visualization
fig.update_traces(textinfo='label+percent',  # Show label and percent
                  textposition=['inside'] + ['outside'] * (len(industry_counts) - 1))  # First inside, others outside


# Show the plot
fig.show()


# ## Percentage Distribution of Job Title Categories by Employment Type
# 

# In[22]:


# Step 1: Group by 'employment_type' and 'job_title_category' to count the number of job postings per combination
job_category_counts = job_postings_df.groupby(['employment_type', 'job_title_category']).size().reset_index(name='count')

# Step 2: Create a pivot table to get employment_type as rows and job_title_category as columns
pivot_table = job_category_counts.pivot_table(index='employment_type', columns='job_title_category', values='count', fill_value=0)

# Step 3: Convert counts to percentages (out of 100) for each employment type
pivot_table_percent = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

# Step 4: Check unique employment types
print("Unique Employment Types:", pivot_table_percent.index.unique())

# Step 5: Sort the columns for both employment types in descending order
sorted_columns = []

# Sort for 'Full-time'
if 'Full-time' in pivot_table_percent.index:
    full_time_sorted = pivot_table_percent.loc['Full-time'].sort_values(ascending=False).index.tolist()
    sorted_columns.extend(full_time_sorted)

# Sort for 'Contract'
if 'Contract' in pivot_table_percent.index:
    contract_sorted = pivot_table_percent.loc['Contract'].sort_values(ascending=False).index.tolist()
    sorted_columns.extend(contract_sorted)

# Remove duplicates while preserving order
sorted_columns = list(dict.fromkeys(sorted_columns))

# Reindex the pivot table based on the sorted columns
pivot_table_percent_sorted = pivot_table_percent[sorted_columns]

# Step 6: Plot the stacked bar plot
ax = pivot_table_percent_sorted.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')

# Step 7: Add titles and labels
plt.title('Percentage Distribution of Job Title Categories by Employment Type')
plt.xlabel('Employment Type')
plt.ylabel('Percentage of Job Postings')

# Step 8: Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()

# Step 9: Add percentage labels inside the bars
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()  # (x,y) = bottom left of the rectangle
    ax.text(x + width / 2, y + height / 2, f'{height:.1f}%', ha='center', va='center')

# Step 10: Show the plot
plt.show()


# ## Average Salary Distribution by Job Title Category - Bar Chart

# In[23]:


# Step 1: Calculate average salaries
job_postings_df['Average_Salary'] = (job_postings_df['Min_Salary'] + job_postings_df['Max_Salary']) / 2

# Step 2: Group by job title category and calculate the average salary
avg_salary_by_category = job_postings_df.groupby('job_title_category')['Average_Salary'].mean().reset_index()

# Step 3: Sort the DataFrame by Average_Salary in descending order
avg_salary_by_category = avg_salary_by_category.sort_values(by='Average_Salary', ascending=False)

# Step 4: Create a color palette (you can use any colormap from matplotlib)
colors = plt.cm.viridis(np.linspace(0, 1, len(avg_salary_by_category)))

# Step 5: Create a bar chart
plt.figure(figsize=(14, 7))
plt.bar(avg_salary_by_category['job_title_category'], avg_salary_by_category['Average_Salary'], color=colors)

# Step 6: Add titles and labels
plt.title('Average Salary by Job Title Category')
plt.xlabel('Job Title Categories')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)

# Step 7: Show the plot
plt.tight_layout()
plt.show()


# ## Average Salary Distribution by Seniority Level

# In[25]:


# Step 1: Calculate average salaries
job_postings_df['Average_Salary'] = (job_postings_df['Min_Salary'] + job_postings_df['Max_Salary']) / 2

# Step 2: Group by seniority to calculate the average salary
avg_salary_by_seniority = job_postings_df.groupby('seniority')['Average_Salary'].mean().reset_index()

# Step 3: Sort the DataFrame in descending order by Average Salary
avg_salary_by_seniority = avg_salary_by_seniority.sort_values(by='Average_Salary', ascending=False)

# Step 4: Create a color palette (using a colormap)
colors = plt.cm.viridis(np.linspace(0, 1, len(avg_salary_by_seniority)))

# Step 5: Create a bar chart with colorful bars
plt.figure(figsize=(14, 7))
plt.bar(avg_salary_by_seniority['seniority'], avg_salary_by_seniority['Average_Salary'], color=colors)

# Step 6: Add titles and labels
plt.title('Average Salary Distribution by Seniority Level')
plt.xlabel('Seniority Levels')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)

# Step 7: Show the plot
plt.tight_layout()
plt.show()


# ## Average Salary Distribution by Industry Category

# In[26]:


# Step 1: Calculate average salaries
job_postings_df['Average_Salary'] = (job_postings_df['Min_Salary'] + job_postings_df['Max_Salary']) / 2

# Step 2: Group by industry_category to calculate the average salary
avg_salary_by_industry = job_postings_df.groupby('industry_category')['Average_Salary'].mean().reset_index()

# Step 3: Sort the DataFrame by Average Salary in descending order
avg_salary_by_industry = avg_salary_by_industry.sort_values(by='Average_Salary', ascending=False)

# Step 4: Create a color palette (using a colormap)
colors = plt.cm.viridis(np.linspace(0, 1, len(avg_salary_by_industry)))

# Step 5: Create a horizontal bar chart
plt.figure(figsize=(14, 7))
plt.barh(avg_salary_by_industry['industry_category'], avg_salary_by_industry['Average_Salary'], color=colors)

# Step 6: Add titles and labels
plt.title('Average Salary Distribution by Industry Category', fontsize=16)

plt.xlabel('Average Salary', fontsize=14)
plt.ylabel('Industry Category', fontsize=14)

# Step 7: Show the plot
plt.tight_layout()
plt.show()


# ## Mapping State Abbreviations to Geographic Coordinates
# 
# This section creates latitude and longitude columns in the job_postings_df DataFrame based on the state abbreviations.
# 

# In[27]:


import pandas as pd
import plotly.express as px

# Example dictionary mapping state names to their coordinates
state_coordinates = {
    'TX': (31.9686, -99.9018),
    'MA': (42.4072, -71.3824),
    'CA': (36.7783, -119.4179),
    'IL': (40.6331, -89.3985),
    'OR': (43.8041, -120.5542),
    'NC': (35.7590, -79.0194),
    'WA': (47.7511, -120.7401),
    'NY': (40.7128, -74.0060),
    'PA': (41.2033, -77.1945),
    'AR': (34.7990, -92.3731),
    'DC': (38.9072, -77.0369),
    'MD': (39.0458, -76.6413),
    'UT': (39.3200, -111.0937)
}

# Creating latitude and longitude columns in your DataFrame
job_postings_df['latitude'] = job_postings_df['State'].map(lambda state: state_coordinates[state][0] if state in state_coordinates else None)
job_postings_df['longitude'] = job_postings_df['State'].map(lambda state: state_coordinates[state][1] if state in state_coordinates else None)


# ## Geographic Distribution of Job Postings by State based on Average Salaries

# In[28]:


# Count the number of job postings per state
job_count_by_state = job_postings_df.groupby('State').size().reset_index(name='Job_Count')

# Merge job count with coordinates
job_count_with_coords = job_postings_df[['State', 'latitude', 'longitude']].drop_duplicates().merge(job_count_by_state, on='State', how='left')

# Step to calculate average salary by state using Max_Salary for a more comprehensive view
avg_salary_by_state = job_postings_df.groupby('State').agg(Average_Salary=('Max_Salary', 'mean')).reset_index()

# Merge average salary with job count and coordinates
avg_salary_with_counts = avg_salary_by_state.merge(job_count_with_coords, on='State', how='outer')  # Use 'outer' to keep all states

# Create the scatter geo plot
fig = px.scatter_geo(avg_salary_with_counts, 
                     lat='latitude', 
                     lon='longitude', 
                     size='Average_Salary',      # Use Average_Salary for marker size
                     color='Average_Salary',     # Use Average_Salary for color
                     projection='albers usa',
                     title='Geographic Distribution of Job Postings by State based on Average Salaries',
                     labels={'Job_Count': 'Number of Job Postings'},  # Update this label if needed
                     opacity=0.7,
                     size_max=30,                # Adjust the maximum size of the markers
                     range_color=[avg_salary_with_counts['Average_Salary'].min(), avg_salary_with_counts['Average_Salary'].max()],  # Define color range
                     text='State'                # Use the state names as labels
                    )

# Update layout for better visibility
fig.update_geos(projection_type="albers usa")
fig.update_layout(coloraxis_colorbar=dict(title='Average Salary ($)'))  # Optional: remove if not needed

# Save the plot as an HTML file
fig.write_html('job_postings_by_state.html')

# Show the plot (if needed)
fig.show()


# ## Salary Insights: Top 10 Companies by Maximum Salary

# In[29]:


job_postings_df['Max_Salary'] = pd.to_numeric(job_postings_df['Max_Salary'], errors='coerce')

# Drop rows with NaN values in the Max_Salary or company_name columns
filtered_df = job_postings_df.dropna(subset=['Max_Salary', 'company_name'])

# Get the top 10 companies with the highest Max_Salary
top_companies = filtered_df[['company_name', 'Max_Salary']].drop_duplicates('company_name').nlargest(10, 'Max_Salary')

# Check if we have enough companies
if top_companies.shape[0] < 10:
    print("Warning: Less than 10 unique companies found.")
else:
    # Define a list of distinct colors
    distinct_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow-green
        '#17becf'   # cyan
    ]

    # Create a bar chart with distinct colors for each bar
    fig = px.bar(top_companies, 
                 x='company_name', 
                 y='Max_Salary', 
                 title='Top 10 Companies by Maximum Salary',
                 labels={'Max_Salary': 'Maximum Salary ($)', 'company_name': 'Company Name'},
                 color='company_name',  # Use company names to assign colors
                 color_discrete_sequence=distinct_colors  # Apply distinct colors from the list
                )

    # Update layout for better visibility
    fig.update_layout(xaxis_title='Company Name', yaxis_title='Maximum Salary ($)', showlegend=False)

    # Show the plot
    fig.show()


# ## Salary Insights: Top 5 Industry Categories Based on Maximum Salary

# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns

# Step 4: Create a box plot with a new color palette
plt.figure(figsize=(14, 7))
sns.boxplot(data=filtered_df, x='industry_category', y='Average_Salary', hue='industry_category', palette='plasma')  # Removed 'legend=False'

# Step 5: Add titles and labels
plt.title('Top 5 Industry Categories Based on Maximum Salary', fontsize=16)
plt.xlabel('Industry Category', fontsize=14)
plt.ylabel('Average Salary', fontsize=14)
plt.xticks(rotation=45)

# Step 6: Adjust legend manually if needed
plt.legend(title="Industry Category")  # Optional: Customize or remove the legend

# Step 7: Show the plot
plt.tight_layout()
plt.show()


# ## Entry Level and Associate Roles – Deep dive
# 
# ### Top 10 Companies Offering Entry Level and Associate Roles

# In[31]:


# Filter for Entry level and Associate roles
filtered_df = job_postings_df[job_postings_df['seniority'].isin(['Entry level', 'Associate'])]

# Group by company_name and count the number of postings
company_counts = filtered_df['company_name'].value_counts().reset_index()
company_counts.columns = ['company_name', 'job_count']

# Select top 10 companies
top_companies = company_counts.nlargest(10, 'job_count')

# Create a pie chart for top companies
fig = px.pie(top_companies, 
             names='company_name', 
             values='job_count', 
             title='Top 10 Companies Offering Entry Level and Associate Roles',
             color_discrete_sequence=distinct_colors  # Use your distinct colors
            )

# Update layout for better visibility
fig.update_traces(textinfo='percent+label')  # Show percentages and labels

# Show the plot
fig.show()


# ## Top 5 Industries Offering Entry Level and Associate Roles

# In[32]:


# Filter for Entry level and Associate roles
filtered_df = job_postings_df[job_postings_df['seniority'].isin(['Entry level', 'Associate'])].copy()  # Create a copy to avoid warning

# Group by industry_category and count the number of postings
industry_counts = filtered_df['industry_category'].value_counts().reset_index()
industry_counts.columns = ['industry_category', 'job_count']

# Sort the results by job count in descending order
sorted_industry_counts = industry_counts.sort_values(by='job_count', ascending=False)

# Select the top 4 or 5 industries
top_industries = sorted_industry_counts.head(5).copy()  # Create a copy to avoid warning

# Calculate the percentage of job counts based on the total number of job postings
total_job_count = top_industries['job_count'].sum()
top_industries.loc[:, 'percentage'] = (top_industries['job_count'] / total_job_count) * 100  # Use .loc to set values

# Define a list of distinct colors for top 4-5 industries
distinct_colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd'   # purple (this will be the 5th color if needed)
]

# Create a bar chart for industry job counts as percentages
fig = px.bar(
    top_industries,
    x='industry_category',
    y='percentage',
    title='Top 5 Industries Offering Entry Level and Associate Roles',
    labels={'percentage': 'Percentage of Job Postings', 'industry_category': 'Industry Category'},
    color='industry_category',  # Use industry category for color
    color_discrete_sequence=distinct_colors[:top_industries.shape[0]]  # Select colors based on the number of top industries
)

# Update layout for better visibility
fig.update_layout(
    xaxis_title='Industry Category',
    yaxis_title='Percentage of Job Postings',
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
    showlegend=False  # Hide the legend since it's not necessary for this chart
)

# Show the plot
fig.show()


# In[ ]:




