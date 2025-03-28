#!/usr/bin/env python
# coding: utf-8

# # **Walkthrough of LinkedIn Job Scraping Code**
# 
# ## Introduction
# This notebook demonstrates how to scrape job postings from LinkedIn using Python libraries such as requests, BeautifulSoup, and pandas. The target job title is "{job_title}" located in {City}.

# ### **1. Import Necessary Libraries**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import random
import pandas as pd


# ### **2. Initialize Job List**

# In[2]:


job_csv = []


# ### **3. Set Search Parameters**

# In[3]:


title = "Data Architect"  # Job title
location = "Los Angeles"  # Job location
start = 1  # Starting point for pagination


# ### **4. Construct URL and Send Request**
# 
# Create the URL for LinkedIn job search, send a GET request, and parse the response to find job postings.

# In[4]:


# Construct the URL for LinkedIn job search
list_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={title}&location={location}&start={start}"

# Send a GET request to the URL and store the response
response = requests.get(list_url)

#Get the HTML, parse the response and find all list items(jobs postings)
list_data = response.text
list_soup = BeautifulSoup(list_data, "html.parser")
page_jobs = list_soup.find_all("li")


# ### **5. Initialize Job ID List**

# In[5]:


id_list = []


# ### **6. Extract Job IDs**

# In[6]:


# Loop through each job posting and extract the job ID
for job in page_jobs:
    # Find the div element containing the job details
    base_card_div = job.find("div", {"class": "base-card"})

    # Extract the job ID from the data-entity-urn attribute
    job_id = base_card_div.get("data-entity-urn").split(":")[3]
    print(job_id)  # Print the job ID for verification

    # Append the job ID to the id_list
    id_list.append(job_id)


# ### **7. Fetch Job Details**

# In[7]:


job_list = []

for job_id in id_list:
    # Construct the URL for each job using the job ID
    job_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"

        # Send a GET request to the job URL and parse the reponse
    job_response = requests.get(job_url)
    print(job_response.status_code)
    job_soup = BeautifulSoup(job_response.text, "html.parser")

     # Create a dictionary to store job details
    job_post = {}

    try:
        job_post["job_title"] = job_soup.find("h2", {"class":"top-card-layout__title font-sans text-lg papabear:text-xl font-bold leading-open text-color-text mb-0 topcard__title"}).text.strip()
    except:
        job_post["job_title"] = None

    # Try to extract and store the company name
    try:
        job_post["company_name"] = job_soup.find("a", {"class": "topcard__org-name-link topcard__flavor--black-link"}).text.strip()
    except:
        job_post["company_name"] = None

    try:
        job_post['location'] = job_soup.find("span", {"class": "topcard__flavor topcard__flavor--bullet"}).text.strip()
    except:
        job_post['location'] = None

    # Try to extract and store the time posted
    try:
        job_post["time_posted"] = job_soup.find("span", {"class": "posted-time-ago__text topcard__flavor--metadata"}).text.strip()
    except:
        job_post["time_posted"] = None

    # Try to extract and store the number of applicants
    try:
        job_post["num_applicants"] = job_soup.find("span", {"class": "num-applicants__caption topcard__flavor--metadata topcard__flavor--bullet"}).text.strip()
    except:
        job_post["num_applicants"] = None

    try:
        job_post["salary"] = job_soup.find("div", {"class": "salary compensation__salary"}).text.strip()
    except:
        job_post["salary"] = None

    try:
        job_post["seniority"] = job_soup.find('li', class_='description__job-criteria-item').find('span').text.strip()
    except:
        job_post["seniority"] = None

    try:
        job_post['employment_type'] = job_soup.find_all('li', class_='description__job-criteria-item')[1].find('span').text.strip()
    except:
        job_post['employment_type'] = None

    try:
        job_post['job_function'] = job_soup.find_all('li', class_='description__job-criteria-item')[2].find('span').text.strip()
    except:
        job_post['job_function'] = None

    try:
        job_post['industry'] = job_soup.find_all('li', class_='description__job-criteria-item')[3].find('span').text.strip()
    except:
        job_post['industry'] = None

    job_list.append(job_post)


# ### **8. Append Job Details to CSV List**

# In[8]:


# Extend the job_csv list with the job details from job_list
job_csv.extend(job_list)


# ### **9. Create DataFrame and Clean Data**

# In[9]:


# Create a DataFrame from the job_csv list
jobs_df = pd.DataFrame(job_csv)

# Drop rows where all elements are missing
jobs_df.dropna(how='all', inplace=True)

# Reset the index of the DataFrame
jobs_df.reset_index(drop=True, inplace=True)

# Drop duplicate rows based on specified columns
jobs_df.drop_duplicates(subset=['job_title', 'company_name', 'location', 'salary', 'seniority', 'employment_type'], keep='first', inplace=True)

# Reset the index of the DataFrame again after dropping duplicates
jobs_df.reset_index(drop=True, inplace=True)

len(jobs_df)


# ### **10. Save Data to CSV**

# In[10]:


# Save the cleaned job data to a CSV file
jobs_df.to_csv('data_architect_52.csv', index=False)


# In[ ]:


df = pd.read_csv('/content/jobposting_data_std.csv')


# In[ ]:


df.head()

