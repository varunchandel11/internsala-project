#!/usr/bin/env python
# coding: utf-8

# In[135]:


get_ipython().system('pip install pandas matplotlib seaborn numpy')


# Instructions												
# Imagine you are working for an organization that offers advanced certifications in various courses. Your objective is to analyze a dataset that contains information about how leads are acquired, categorized and converted.												
# Given the dataset (attached), which includes the following columns:												
# 1. Channel_group: Acquisition channel through which the lead was generated.												
# 2. Course: The course the lead showed interest in.												
# 3. Lead_id: A unique identifier for each lead.												
# 4. Lead_type: The type of interaction through which the lead was generated.												
# 5. Lead_date: The date when the lead was created												
# 6. Other intuitive columns…..												
# You need to:												
# 1. Identify the top 5 insights from this dataset.												
# 2. Present your analysis in a clear and structured manner in a document(2-4 pages) using either a Word document or a PDF format.												
# 3. Make clear assumptions if necessary.												
# Your analysis should highlight trends, patterns, or any other meaningful observations that could help the organization understand how leads are generated and how they interact with courses.												
# 												

# # Step 1: Import Libraries

# In[170]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
import streamlit as st
import plotly.express as px


# In[171]:


# Create a copy of original data (best practice)
df_original = pd.read_csv('SDA_assignment .csv')
df = df_original.copy()


# # Step 2: Load and Preview the Dataset

# In[172]:


# Load the dataset
data = pd.read_csv('SDA_assignment .csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head(10))

# Get basic information about the dataset
print("\nDataset Information:")
print(data.info())


# In[173]:


# Check for missing values in each column
missing_values = data.isnull().sum()
print("\nMissing Values in the Dataset:")
print(missing_values)


# In[174]:


# Print column names to ensure they match
print(data.columns)


# # Step 3: Clean Column Names (Remove Leading/Trailing Spaces)

# In[175]:


# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Display updated column names
print("Updated Column Names:")
print(data.columns)


# # Step 4: Handle Missing Values
# Here, we'll check for missing values in the dataset and handle them:
# 
# Check for missing values in each column.
# 
# 1 : Fill missing amount_paid values with 0.
# 
# 2 : Fill missing paid_at values with 'Not Paid'.
# 
# 3 : Drop rows with missing lead_id since it's a unique identifier.

# In[176]:


# Check for missing values in each column
missing_values = data.isnull().sum()
print("\nMissing Values in the Dataset:")
print(missing_values)

# Fill missing 'amount_paid' with 0 and 'paid_at' with 'Not Paid'
data['amount_paid'].fillna(0, inplace=True)
data['paid_at'].fillna('Not Paid', inplace=True)

# Drop rows with missing 'lead_id' since it's a unique identifier
data.dropna(subset=['lead_id'], inplace=True)

# Verify the data after handling missing values
print("\nDataset After Handling Missing Values:")
print(data.head(10))


# # Step 5: Convert lead_date to Datetime Format
#  convert the lead_date column to a proper datetime format to make it easier for analysis and time-based operations.

# In[177]:


# Convert 'lead_date' to datetime format
data['lead_date'] = pd.to_datetime(data['lead_date'], format='%d-%m-%Y %H:%M')

# Verify the conversion
print("\nDataset with 'lead_date' as Datetime:")
print(data.head(10))


# # Step 6: Analyze the Data for Insights
# Let's now proceed to analyzing the dataset to identify insights. A few steps to start with:
# 
# 1 : Top 5 Acquisition Channels by Lead Count: We can analyze which acquisition channels bring in the most leads.

# In[178]:


# Top 5 channels by lead count
top_channels = data['Channel_group'].value_counts().head(5)
print("\nTop 5 Acquisition Channels by Lead Count:")
print(top_channels)


# In[179]:


# Visualization: Top 5 Acquisition Channels in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

top_channels_values = top_channels.values
top_channels_indices = range(len(top_channels))
colors = cm.viridis(np.linspace(0, 1, len(top_channels)))

# Create 3D bars with a colormap applied
ax.bar(top_channels_indices, top_channels_values, zs=0, zdir='y', color=colors, alpha=0.8)
ax.set_xticks(top_channels_indices)
ax.set_xticklabels(top_channels.index, rotation=45)
ax.set_xlabel('Channels')
ax.set_ylabel('Z-axis')
ax.set_zlabel('Counts')
ax.set_title('Top Acquisition Channels (3D Bar Chart)')

plt.show()


# In[180]:


# Analysis and Visualization 1: Top 5 Acquisition Channels
plt.figure(figsize=(12, 6))
top_channels = data['Channel_group'].value_counts().head(5)
sns.barplot(x=top_channels.index, y=top_channels.values)
plt.title('Top 5 Acquisition Channels by Lead Count', fontsize=14)
plt.xlabel('Channel Group', fontsize=12)
plt.ylabel('Number of Leads', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 2: Top 5 Courses by Lead Interest: Find which courses are most popular among the leads

# In[181]:


# Top 5 courses by lead count
top_courses = data['course'].value_counts().head(5)
print("\nTop 5 Courses by Lead Interest:")
print(top_courses)


# In[182]:



fig = plt.figure(figsize=(12, 8))  # Increased figure size
ax = fig.add_subplot(111, projection='3d')

# Data for the bar chart
top_courses_values = top_courses.values  # Heights of the bars
top_courses_indices = np.arange(len(top_courses))  # Indices for the bars

# Use a colormap to generate colors for the bars
cmap = cm.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, len(top_courses_values)))

# Create the 3D bar chart
ax.bar(top_courses_indices, top_courses_values, zs=0, zdir='y', color=colors, alpha=0.8)

# Set axis labels
ax.set_xlabel('Courses', fontsize=12)
ax.set_ylabel('Y-axis (placeholder)', fontsize=12)
ax.set_zlabel('Lead Counts', fontsize=12)
ax.set_xticks(top_courses_indices)
ax.set_xticklabels(top_courses.index, rotation=45, fontsize=10)  # Reduced font size

plt.title('Top Courses by Lead Counts (3D View)', fontsize=14)

# Adjust margins manually to avoid overlap
plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.3)

plt.show()


# In[183]:


# Analysis and Visualization 2: Top 5 Courses
plt.figure(figsize=(12, 6))
top_courses = data['course'].value_counts().head(5)
sns.barplot(x=top_courses.index, y=top_courses.values)
plt.title('Top 5 Courses by Lead Interest', fontsize=14)
plt.xlabel('Course Name', fontsize=12)
plt.ylabel('Number of Leads', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 3 : Distribution of Leads Over Time: Analyze the number of leads generated over time to identify trends or patterns.

# In[184]:


# Number of leads over time
leads_over_time = data['lead_date'].groupby(data['lead_date'].dt.to_period('M')).count()
print("\nNumber of Leads Over Time:")
print(leads_over_time)


# In[185]:


# Visualization: Monthly Lead Trends in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

months = range(len(monthly_leads))
leads = monthly_leads.values

ax.plot(months, leads, zs=0, zdir='y', marker='o', color='b')
ax.set_xticks(months)
ax.set_xticklabels(monthly_leads.index.strftime('%Y-%m'), rotation=45)
ax.set_title('Lead Generation Over Time (3D)')
ax.set_xlabel('Month')
ax.set_ylabel('Index')
ax.set_zlabel('Number of Leads')
plt.show()


# In[186]:


# Visualization: Monthly Lead Trends
plt.figure(figsize=(12, 6))
monthly_leads.plot(kind='line', marker='o', color='b')
plt.title('Lead Generation Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Leads')
plt.grid()
plt.show()


# In[187]:


# Insight 4: Lead Types and Their Effectiveness
top_lead_types = data['Lead_type'].value_counts().head(5)
print("\nTop 5 Lead Types by Count:")
print(top_lead_types)


# In[188]:


# Visualization: Top Lead Types in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

top_lead_types_values = top_lead_types.values
top_lead_types_indices = range(len(top_lead_types))

# Use a valid color, such as 'lightblue', or generate a list of colors
ax.bar(top_lead_types_indices, top_lead_types_values, zs=0, zdir='y', color='lightblue', alpha=0.9)

ax.set_xticks(top_lead_types_indices)
ax.set_xticklabels(top_lead_types.index, rotation=45)
ax.set_title('Top 5 Lead Types by Count (3D)')
ax.set_xlabel('Lead Type')
ax.set_ylabel('Index')
ax.set_zlabel('Number of Leads')
plt.show()


# In[189]:


# Visualization: Top Lead Types
plt.figure(figsize=(12, 6))
sns.barplot(x=top_lead_types.index, y=top_lead_types.values, palette='pastel')
plt.title('Top 5 Lead Types by Count')
plt.xlabel('Lead Type')
plt.ylabel('Number of Leads')
plt.xticks(rotation=45)
plt.show()


# In[190]:



# Analysis and Visualization 3: Lead Type Distribution
plt.figure(figsize=(10, 6))
lead_type_dist = data['Lead_type'].value_counts()
plt.pie(lead_type_dist.values, labels=lead_type_dist.index, autopct='%1.1f%%')
plt.title('Distribution of Lead Types', fontsize=14)
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[191]:


# Analysis and Visualization 4: Daily Lead Generation Trend
plt.figure(figsize=(15, 6))
daily_leads = data.groupby(data['lead_date'].dt.date)['lead_id'].count()
plt.plot(daily_leads.index, daily_leads.values, marker='o')
plt.title('Daily Lead Generation Trend', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Leads', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[192]:


# Analysis and Visualization 5: Graduation Year Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=data, x='graduation_year', bins=20)
plt.title('Distribution of Leads by Graduation Year', fontsize=14)
plt.xlabel('Graduation Year', fontsize=12)
plt.ylabel('Number of Leads', fontsize=12)
plt.tight_layout()
plt.show()


# In[193]:


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

top_lead_types_values = top_lead_types.values
top_lead_types_indices = range(len(top_lead_types))

# Pass a list of colors (one for each bar)
colors = ['lightblue'] * len(top_lead_types_values)
ax.bar(top_lead_types_indices, top_lead_types_values, zs=0, zdir='y', color=colors, alpha=0.8)

ax.set_xticks(top_lead_types_indices)
ax.set_xticklabels(top_lead_types.index, rotation=45)
ax.set_title('Top 5 Lead Types by Count (3D)')
ax.set_xlabel('Lead Type')
ax.set_ylabel


# In[194]:


# Print key insights
print("\nKey Insights:")
print("\n1. Top 5 Acquisition Channels by Lead Count:")
print(top_channels)

print("\n2. Top 5 Courses by Lead Interest:")
print(top_courses)

print("\n3. Lead Type Distribution:")
print(data['Lead_type'].value_counts())


# In[195]:


# Step 4: Lead Conversion Analysis
# Define conversion: leads with amount_paid > 0 or paid_at != 'Not Paid'
data['converted'] = np.where((data['amount_paid'] > 0) | (data['paid_at'] != 'Not Paid'), 1, 0)


# In[196]:


# Conversion Rate by Channel
conversion_rate_by_channel = data.groupby('Channel_group')['converted'].mean().sort_values(ascending=False)
print("\nConversion Rate by Channel:")
print(conversion_rate_by_channel)


# In[197]:


# Visualization: Conversion Rate by Channel
plt.figure(figsize=(12, 6))
sns.barplot(x=conversion_rate_by_channel.index, y=conversion_rate_by_channel.values, palette='viridis')
plt.title('Conversion Rate by Channel')
plt.xlabel('Channel Group')
plt.ylabel('Conversion Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[198]:


# Conversion Rate by Course
conversion_rate_by_course = data.groupby('course')['converted'].mean().sort_values(ascending=False)
print("\nConversion Rate by Course:")
print(conversion_rate_by_course)


# In[199]:


# Visualization: Conversion Rate by Course
plt.figure(figsize=(12, 6))
sns.barplot(x=conversion_rate_by_course.index, y=conversion_rate_by_course.values, palette='coolwarm')
plt.title('Conversion Rate by Course')
plt.xlabel('Course')
plt.ylabel('Conversion Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[200]:


# Step 5: Assumptions Clarification
# Explicitly state assumptions for the analysis
assumptions = [
    "Missing values in 'amount_paid' were filled with 0, assuming they represent unpaid leads.",
    "Missing values in 'paid_at' were replaced with 'Not Paid', indicating no payment was made.",
    "Rows with missing 'lead_id' were dropped as 'lead_id' is a unique identifier.",
    "Conversion was defined as having a positive 'amount_paid' or a value in 'paid_at' other than 'Not Paid'."
]

print("\nAssumptions:")
for assumption in assumptions:
    print(f"- {assumption}")


# In[201]:


# Final Insights and Recommendations
print("\nKey Insights:")
print("\n1. Monthly Lead Trends:")
print(monthly_leads)
print("\n2. Conversion Rate by Channel:")
print(conversion_rate_by_channel)
print("\n3. Conversion Rate by Course:")
print(conversion_rate_by_course)

recommendations = [
    "Focus marketing efforts on channels with high conversion rates.",
    "Promote courses with higher conversion rates to maximize enrollments.",
    "Analyze why certain channels/courses have low conversions and optimize strategies accordingly."
]

print("\nRecommendations:")
for recommendation in recommendations:
    print(f"- {recommendation}")


# In[169]:


pip install streamlit plotly pandas


# In[202]:


# Sidebar Filters
st.sidebar.header("Filters")
selected_channel = st.sidebar.multiselect("Select Channels", options=data['Channel_group'].unique(), default=data['Channel_group'].unique())
selected_course = st.sidebar.multiselect("Select Courses", options=data['course'].unique(), default=data['course'].unique())

filtered_data = data[(data['Channel_group'].isin(selected_channel)) & (data['course'].isin(selected_course))]

# Dashboard Title
st.title("Lead Analysis Dashboard")

# 1. Lead Trends Over Time
st.subheader("Lead Generation Over Time")
leads_over_time = filtered_data.groupby(filtered_data['lead_date'].dt.to_period('M')).size()
fig1 = px.line(leads_over_time, x=leads_over_time.index.astype(str), y=leads_over_time.values, labels={'x': 'Month', 'y': 'Number of Leads'})
st.plotly_chart(fig1)

# 2. Top Channels
st.subheader("Top Acquisition Channels")
channel_counts = filtered_data['Channel_group'].value_counts().head(5)
fig2 = px.bar(channel_counts, x=channel_counts.index, y=channel_counts.values, labels={'x': 'Channel', 'y': 'Lead Count'})
st.plotly_chart(fig2)

# 3. Top Courses
st.subheader("Top Courses")
course_counts = filtered_data['course'].value_counts().head(5)
fig3 = px.bar(course_counts, x=course_counts.index, y=course_counts.values, labels={'x': 'Course', 'y': 'Lead Count'})
st.plotly_chart(fig3)

# 4. Conversion Rate by Channel
st.subheader("Conversion Rate by Channel")
conversion_rate_by_channel = filtered_data.groupby('Channel_group')['converted'].mean().sort_values(ascending=False)
fig4 = px.bar(conversion_rate_by_channel, x=conversion_rate_by_channel.index, y=conversion_rate_by_channel.values, labels={'x': 'Channel', 'y': 'Conversion Rate'})
st.plotly_chart(fig4)

# 5. Conversion Rate by Course
st.subheader("Conversion Rate by Course")
conversion_rate_by_course = filtered_data.groupby('course')['converted'].mean().sort_values(ascending=False)
fig5 = px.bar(conversion_rate_by_course, x=conversion_rate_by_course.index, y=conversion_rate_by_course.values, labels={'x': 'Course', 'y': 'Conversion Rate'})
st.plotly_chart(fig5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Key Insights:
# Top 5 Acquisition Channels by Lead Count:
# 
# The channel with the highest number of leads is A, contributing 7932 leads.
# Other notable channels include M (1647), F (1586), D (1294), and E (1080).
# Top 5 Courses by Lead Interest:
# 
# Python is the most popular course with 4323 leads.
# Other highly popular courses include Java (4250 leads), CRM (2565 leads), Guitar (2164 leads), and Google Analytics (1358 leads).
# Leads Generated Over Time:
# 
# All the data points fall within May 2024, indicating that leads were generated consistently in this time period, totaling 16,460 leads in the month.

# 
# ### **Data Analysis Report: Lead Acquisition and Course Interest**
# 
# #### **1. Introduction**
# The purpose of this analysis is to explore how leads are acquired and interacted with various courses in an organization offering advanced certifications. The dataset provided contains information about how leads were generated, categorized, and converted, offering a comprehensive view of the lead acquisition process.
# 
# #### **2. Data Overview**
# The dataset consists of **16,460 entries** and **8 columns**, including:
# - **Channel_group**: The acquisition channel through which the lead was generated.
# - **Course**: The course in which the lead showed interest.
# - **Lead_id**: A unique identifier for each lead.
# - **Lead_type**: The type of interaction through which the lead was generated.
# - **Lead_date**: The date when the lead was created.
# - **Graduation_year**: The year the lead graduated.
# - **Amount_paid**: The amount paid by the lead (if applicable).
# - **Paid_at**: The payment status of the lead.
# 
# The dataset was carefully cleaned and preprocessed to handle missing values, and the **'lead_date'** column was converted to a proper `datetime` format to facilitate time-based analysis.
# 
# #### **3. Data Cleaning & Preprocessing**
# - **Missing Values**: Columns like **amount_paid** and **paid_at** had a significant number of missing values. These were handled by filling missing values for **amount_paid** with 0 and setting **paid_at** to 'Not Paid' where missing.
#   
# - **Date Conversion**: The **lead_date** column was converted from string format to `datetime`, enabling accurate temporal analysis.
# 
# #### **4. Top Insights from the Dataset**
# Here are the top insights based on the analysis:
# 
# - **Top 5 Acquisition Channels by Lead Count**:
#    - Channel **A** generated the highest number of leads (7932 leads), followed by channels **M** (1647 leads), **F** (1586 leads), **D** (1294 leads), and **E** (1080 leads). This shows that channel A is by far the most successful in acquiring leads.
# 
# - **Top 5 Courses by Lead Interest**:
#    - The most popular course is **Python** (4323 leads), followed closely by **Java** (4250 leads). Other popular courses include **CRM** (2565 leads), **Guitar** (2164 leads), and **Google Analytics** (1358 leads). This indicates a strong interest in programming courses like Python and Java.
# 
# - **Leads Generated Over Time**:
#    - The data spans **May 2024**, with a total of **16,460 leads** generated during this month. This highlights a consistent flow of leads during this period.
# 
# #### **5. Conclusion**
# The analysis reveals key patterns in how leads are acquired and which courses generate the most interest. 
# - The **A channel** is highly effective for lead generation, and courses like **Python** and **Java** dominate the interest among leads. 
# - It is important for the organization to further focus on these successful channels and courses for future marketing and lead-generation campaigns.
# 
# Based on these insights, I recommend that the organization:
# - Focus marketing efforts on **Channel A** for lead acquisition.
# - Invest in **Python** and **Java** courses as they attract the most interest.
# 
# ---
# 

# In[ ]:




