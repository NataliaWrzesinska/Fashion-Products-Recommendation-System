#!/usr/bin/env python
# coding: utf-8

# # Fashion Products Recommendation System for Sales Enhancement
# 
# ## Project Objective
# The primary goal of this project is to develop a sophisticated recommendation system aimed at boosting sales by offering personalized product suggestions to customers. By leveraging historical data on product interactions, ratings, and user preferences, the system will identify and recommend products that are most likely to appeal to individual customers, thereby enhancing the sales of various fashion products.
# 
# ## Research Questions
# 1. Which product categories show the highest potential for sales increase through personalized recommendations?
# 2. How does the relationship between product price and user ratings impact sales potential?
# 3. What are the key features of products that significantly influence their likelihood of being purchased when recommended?
# 
# ## Hypothesis
# We hypothesize that a well-designed recommendation system can significantly increase the sales of fashion products by aligning product offerings with the unique preferences and behaviors of individual customers.
# 

# ## Loading the Dataset
# 
# The first step in our analysis is to load the dataset. We'll be using a dataset that contains information on various fashion products, including details such as product name, brand, category, price, and rating. This initial loading step is crucial as it sets the stage for all our subsequent data exploration and analysis efforts.
# 
# Let's load the dataset using Pandas, which is a powerful Python library for data manipulation and analysis. We'll then take a quick look at the first few entries in the dataset to get a preliminary understanding of the data structure and the types of information available to us. Following that, we will use the `.info()` method to get a summary of the dataset, including the number of entries, the type of data in each column, and a check for missing values.
# 

# In[6]:


import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\hp\Desktop\Fashion\fashion_products.csv')

# Quick look at the data
df.head()

# Get an overview of the dataset
df.info()


# In[ ]:





# ## Data Cleaning Process
# 
# Before diving into the exploratory data analysis, it's crucial to prepare the dataset to ensure the quality and consistency of the data. This data cleaning process involves several steps:
# 
# 1. **Removing Duplicate Entries**: Duplicates can skew our analysis, so we'll remove any duplicate rows to ensure each entry is unique.
# 
# 2. **Handling Missing Values**: Missing data can also affect our analysis. Depending on the nature of the data and the missing values, we may decide to remove rows with any missing data to maintain the integrity of our dataset.
# 
# 3. **Normalizing Text Data**: To ensure consistency across our textual data, we will convert all text to lowercase. This step is essential for categorical analysis and helps prevent the same category from being interpreted as different due to case differences.
# 
# By performing these data cleaning steps, we aim to create a reliable foundation for our exploratory data analysis and subsequent modeling.
# 

# In[7]:


# Removing duplicate entries
df.drop_duplicates(inplace=True)

# Handling missing values - for example, by removing rows with any missing data
df.dropna(inplace=True)

# Normalizing text data to ensure consistency (e.g., converting all to lowercase)
df['Product Name'] = df['Product Name'].str.lower()
df['Brand'] = df['Brand'].str.lower()
df['Category'] = df['Category'].str.lower()

# Verify the cleaning process
df.info()


# # Exploratory Data Analysis (EDA)
# 
# In this section, we will conduct an exploratory data analysis (EDA) on the fashion products dataset. The goal of EDA is to:
# - Understand the data's underlying structure and characteristics.
# - Identify any anomalies or outliers that may require further investigation.
# - Discover patterns and relationships between variables that can inform our recommendation system.
# 
# We will start with basic statistical analyses, followed by visualizations of product distributions, category popularity, price distributions, and average ratings across different categories. This will provide us with valuable insights into the dataset and help guide our subsequent steps in developing a robust recommendation system.
# 

# In[12]:


# Import necessary libraries for data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the visualization style
sns.set(style="whitegrid")

# Load the dataset
df = pd.read_csv(r'C:\Users\hp\Desktop\Fashion\fashion_products.csv')

# Basic statistics overview
print("Basic Statistics:")
display(df.describe())

# Visualization of Price Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=30, kde=True)
plt.title('Distribution of Product Prices')
plt.xlabel('Price')
plt.ylabel('Number of Products')
plt.show()

# Visualization of Product Category Popularity
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='Category')
plt.title('Popularity of Product Categories')
plt.xlabel('Number of Products')
plt.ylabel('Category')
plt.show()

# Visualization of Average Ratings by Category
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Rating', y='Category', estimator=np.mean, errorbar=None)
plt.title('Average Product Ratings Across Different Categories')
plt.xlabel('Average Rating')
plt.ylabel('Category')
plt.show()

# Visualization of Number of Products by Brand
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='Brand', order=df['Brand'].value_counts().index)
plt.title('Number of Products by Brand')
plt.xlabel('Number of Products')
plt.ylabel('Brand')
plt.show()


# # Building a User-Based Collaborative Filtering Recommendation System
# In this section, we aim to develop a simple yet effective recommendation system using the User-Based Collaborative Filtering approach. This method leverages user ratings to identify similarities between users and recommend products liked by similar users. The process consists of several key steps:
# 
# ## Step 1: Data Preparation
# We start with a DataFrame df containing three key pieces of information for each interaction: 'User ID', 'Product ID', and 'Rating'. Our goal is to transform this data into a user-product matrix where rows represent users, columns represent products, and cell values are the ratings given by users to products.
# 
# ## Step 2: Standardization of Ratings
# To normalize the ratings across different users, we employ StandardScaler from scikit-learn. This ensures that our similarity calculations are not biased by users who tend to give higher or lower ratings in general.
# 
# ## Step 3: Calculating Cosine Similarity
# We calculate the cosine similarity among users based on their rating patterns. Cosine similarity measures the cosine of the angle between two vectors, in this case, the rating vectors of two users. A value closer to 1 indicates a higher similarity.
# 
# ## Step 4: Generating Recommendations
# With the similarity matrix in hand, we can now generate recommendations. For a given user, we identify similar users and recommend products that these similar users have rated highly but the given user has not yet rated. This approach assumes that users with similar tastes are likely to enjoy similar products.
# 
# ## Implementation:
# Below is the Python code that implements the above steps. It includes functions to create a pivot table from the original DataFrame, standardize the ratings, calculate user-user cosine similarity, and finally, recommend products to a given user based on the similarities.

# In[26]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Assuming 'df' is your DataFrame containing 'User ID', 'Product ID', and 'Rating'

# Create a pivot table
pivot_table = df.pivot(index='User ID', columns='Product ID', values='Rating').fillna(0)

# Standardize the ratings
scaler = StandardScaler()
pivot_table_scaled = scaler.fit_transform(pivot_table)

# Calculate cosine similarity among users
similarity_matrix = cosine_similarity(pivot_table_scaled)

# Convert the similarity matrix to a DataFrame for better readability
similarity_df = pd.DataFrame(similarity_matrix, index=pivot_table.index, columns=pivot_table.index)

def recommend_products(user_id, similarity_df, pivot_table, n_recommendations=5):
    """
    Recommend products to a user based on user-user similarity.
    
    Parameters:
    - user_id: The ID of the user to whom recommendations should be made.
    - similarity_df: DataFrame containing user-user similarity scores.
    - pivot_table: The original pivot table of users and product ratings.
    - n_recommendations: Number of recommendations to return.
    
    Returns:
    - A list of recommended product IDs.
    """
    # Get the top similar users to the user_id
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:n_recommendations+1].index
    
    # Get the products rated by these similar users
    recommended_products = pivot_table.loc[similar_users].mean().sort_values(ascending=False).index.tolist()
    
    # Filter out products the user has already rated
    rated_products = pivot_table.loc[user_id][pivot_table.loc[user_id]>0].index
    recommendations = [product for product in recommended_products if product not in rated_products]
    
    return recommendations[:n_recommendations]

# Example: Recommend products for user with ID 1
recommended_product_ids = recommend_products(user_id=1, similarity_df=similarity_df, pivot_table=pivot_table, n_recommendations=5)
print(f"Recommended Product IDs for User 1: {recommended_product_ids}")


# # Simple Recommendation Model Evaluation
# In this section, we evaluate a straightforward recommendation model based on average product ratings. Our dataset consists of user-product interactions, identified by 'User ID' and 'Product ID', along with the ratings given. We predict potential ratings using the average rating for each product, a basic yet insightful approach for generating recommendations.
# 
# ## Process Overview:
# Prediction Generation: We calculate each product's average rating and use these averages as predicted ratings for all users.
# Evaluation Metrics: To assess our model, we focus on precision and recall at 'k'. These metrics help us understand the relevance of our top 'k' recommendations, providing a measure of our model's effectiveness.
# This streamlined analysis will guide us through generating predictions and evaluating their accuracy, offering a foundation for more complex recommendation systems.

# In[31]:


import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Assume 'df' is your DataFrame that you've loaded from a CSV file or another data source.
# It contains 'User ID', 'Product ID', and 'Rating' columns representing the user-item interactions.

# Preparing a list of predictions
# Here, we're generating average ratings for each product based on existing user ratings.
# This simplistic approach uses the mean rating of each product as its estimated rating for all users.
average_ratings = df.groupby('Product ID')['Rating'].mean().to_dict()

predictions = []
# Iterating through each row in the DataFrame to construct a list of predictions.
# For each user-item pair, we're assuming the estimated rating is the average rating of the item.
# This is a naive approach and serves as a baseline for more sophisticated models.
for index, row in df.iterrows():
    user_id = row['User ID']
    product_id = row['Product ID']
    true_rating = row['Rating']
    estimated_rating = average_ratings.get(product_id, 0)  # Default to 0 if no rating is found
    predictions.append((user_id, product_id, true_rating, estimated_rating))

# Function to calculate precision and recall at k
# This function evaluates the performance of our simple recommendation model.
# It calculates precision and recall for each user based on the top-k recommendations
# and aggregates these metrics across all users to find the mean precision and recall.
def calculate_precision_recall_at_k(predictions, k=10, threshold=3.5):
    # Organizing predictions by user
    user_est_true = defaultdict(list)
    for uid, _, true_r, est in predictions:  # Simplified tuple unpacking (removed unused element)
        user_est_true[uid].append((est, true_r))
    
    # Calculating precision and recall
    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sorting user ratings by estimated value to simulate top-k recommendations
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Calculating relevant metrics
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

# Calculating and displaying mean precision and recall at k
precisions, recalls = calculate_precision_recall_at_k(predictions, k=5)
mean_precision = sum(prec for prec in precisions.values()) / len(precisions)
mean_recall = sum(rec for rec in recalls.values()) / len(recalls)

print(f'Mean Precision@5: {mean_precision:.3f}')
print(f'Mean Recall@5: {mean_recall:.3f}')


# ## Visualization 1: Top N Recommended Products for a Selected User
# 

# In[34]:


# Retrieve top N recommendations for user with ID 1
recommended_product_ids = recommend_products(user_id=1, similarity_df=similarity_df, pivot_table=pivot_table, n_recommendations=5)

# Calculate average ratings for recommended products
recommended_ratings = average_ratings[recommended_product_ids]

# Creating the plot
plt.figure(figsize=(10, 6))
sns.barplot(x=recommended_ratings.index, y=recommended_ratings.values)
plt.title('Top 5 Recommended Products for User 1')
plt.xlabel('Product ID')
plt.ylabel('Average Rating')
plt.show()



# ## Visualization 2: Heatmap of User Similarity
# This heatmap allows us to visualize the similarity between users, identifying potential clusters of users with similar tastes.

# In[36]:


# Assuming 'similarity_matrix' was generated earlier
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix[:20, :20], annot=True, cmap='coolwarm')
plt.title('Heatmap of Similarity Among the First 20 Users')
plt.xlabel('User Index')
plt.ylabel('User Index')
plt.show()


# ## Visualisation 3: Comparison of Number of Recommendations to Actual Product Ratings
# This visualization can help understand if the recommendation system favors products with a higher number of ratings over less-known ones.

# In[38]:


# Counting the number of ratings for each product
ratings_count = df.groupby('Product ID')['Rating'].count()

# Preparing data for visualization
recommended_counts = ratings_count.loc[recommended_product_ids]  # Assuming `recommended_product_ids` contains IDs of recommended products

# Creating the plot
plt.figure(figsize=(10, 6))
sns.barplot(x=recommended_counts.index, y=recommended_counts.values)
plt.title('Number of Ratings for Recommended Products')
plt.xlabel('Product ID')
plt.ylabel('Number of Ratings')
plt.show()


# ## Visualisation 4: Rating Distribution Visualization for Recommended Products
# This visualization assesses if recommended products are generally highly rated by all users.

# In[39]:


# Preparing data
recommended_ratings_distribution = df[df['Product ID'].isin(recommended_product_ids)]['Rating']

# Creating the plot
plt.figure(figsize=(10, 6))
sns.histplot(recommended_ratings_distribution, bins=10, kde=True)
plt.title('Rating Distribution for Recommended Products')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# ## Visualisation 4: Heatmap of Product Ratings
# Shows how different users rated recommended products, which can help identify products with diverse opinions.

# In[40]:


# Preparing rating matrix for recommended products
recommended_products_matrix = pivot_table[recommended_product_ids]

# Creating heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(recommended_products_matrix, cmap='viridis')
plt.title('Heatmap of Ratings for Recommended Products')
plt.xlabel('Product ID')
plt.ylabel('User ID')
plt.show()


# ## Visualisation 5: Comparison of Precision and Recall for Different Values of K
# Analyzes how precision and recall metrics change depending on the number of recommended products (k).

# In[41]:


ks = range(1, 11)
precisions_at_k = []
recalls_at_k = []

# Calculating precision and recall for different k
for k in ks:
    precisions, recalls = calculate_precision_recall_at_k(predictions, k=k)
    precisions_at_k.append(sum(prec for prec in precisions.values()) / len(precisions))
    recalls_at_k.append(sum(rec for rec in recalls.values()) / len(recalls))

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(ks, precisions_at_k, label='Precision')
plt.plot(ks, recalls_at_k, label='Recall')
plt.xlabel('k')
plt.ylabel('Score')
plt.title('Precision and Recall at Different k Values')
plt.legend()
plt.show()


# # Analysis of Visualizations
# 
# ## Top 5 Recommended Products for User 1
# The bar chart displaying the top 5 recommended products for User 1 shows the average rating for each product. This visualization suggests that the recommendation algorithm favors products with higher average ratings, which could be a sign of the system's effectiveness. However, it is essential to ensure that the model does not exhibit bias towards products with more ratings or higher visibility.
# 
# ## Heatmap of User Similarity
# The heatmap of user similarity indicates how users compare with each other based on their product rating patterns. A high concentration of red and orange suggests strong similarity among certain users, which is beneficial for collaborative filtering. However, the presence of blue and purple squares indicates dissimilar rating behaviors, which may require further investigation or the inclusion of more personalized recommendations.
# 
# ## Number of Ratings for Recommended Products
# This chart compares the number of ratings that recommended products have received. If the recommended products have a significantly higher number of ratings, it may indicate a popularity bias. It's crucial for the recommendation system to balance well-known products with novel recommendations to ensure diversity.
# 
# ## Rating Distribution for Recommended Products
# The distribution of ratings for recommended products, shown as a histogram with a Kernel Density Estimate (KDE) line, suggests that recommended products tend to have a tight cluster of high ratings. This could mean that the system successfully identifies top-rated products but may also benefit from incorporating a range of products to cater to diverse user preferences.
# 
# ## Heatmap of Ratings for Recommended Products
# The heatmap shows user ratings for the recommended products. Yellow lines represent higher ratings, and the prevalence of dark cells may suggest a lack of ratings for some products or users. This could indicate the need for a more robust data collection process or the introduction of mechanisms to encourage user engagement and feedback.
# 
# ## Precision and Recall at Different k Values
# The precision and recall curve provides insight into the trade-off between the two metrics at different values of 'k'. The gradual plateauing of the curve suggests that increasing the number of recommendations may not significantly improve recall. Optimal 'k' values need to be determined to balance both precision and recall effectively.
# 
# # Conclusion
# In conclusion, the visual analysis has provided valuable insights into the behavior of the recommendation system. While there is evidence of the system's ability to identify products that align with user preferences, there is room for improvement in diversity and personalization. Future work should focus on enhancing the recommendation engine's robustness, ensuring that it serves the users' varied interests and uncovers the hidden gems within the product catalog. By continuously iterating on the model and incorporating user feedback, we can aspire to create a dynamic and responsive recommendation system that grows with its user base.

# ## Category Analysis in Product Recommendations
# This script, titled explores the distribution of product categories within a set of recommended items. It merges product recommendations with detailed information to assess category prevalence, displaying the results through a bar chart. This analysis aids in understanding which categories are most frequently recommended, offering insights into consumer preferences and potential sales opportunities.

# In[43]:


# Analyzing Category Recommendations
def analyze_category_recommendations(recommendations, df):
    # Merge recommendations with product details to get categories
    recommended_products = df[df['Product ID'].isin(recommendations)]
    category_counts = recommended_products['Category'].value_counts()
    
    # Plotting the frequency of each category in recommendations
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title('Frequency of Product Categories in Recommendations')
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

    return category_counts

# Assuming 'recommended_product_ids' contains the IDs of recommended products
category_analysis = analyze_category_recommendations(recommended_product_ids, df)
print(category_analysis)


# ## Insights into Product Category Performance
# This script segment conducts an analysis of product categories by aggregating average ratings and the count of recommendations per category. It aims to identify categories with the highest potential for sales increase based on their popularity and customer satisfaction. The results are sorted to prioritize categories with higher average ratings and more recommendations, offering a strategic view for targeted marketing and stock optimization.

# In[42]:


# Group by 'Category' and calculate average rating and recommendation count
category_analysis = df.groupby('Category').agg({
    'Rating': 'mean',
    'Product ID': 'count'  # Assuming 'Product ID' count as a proxy for recommendations
}).rename(columns={'Rating': 'Average Rating', 'Product ID': 'Recommendation Count'})

# Sort categories by Average Rating and Recommendation Count for potential sales increase insight
category_analysis.sort_values(by=['Average Rating', 'Recommendation Count'], ascending=False, inplace=True)

# Display the top categories
print(category_analysis.head())


# ## Color Impact Analysis on Ratings
# This script segment explores the influence of color on product ratings within a fashion products dataset. By grouping products by color and calculating the average rating for each group, it visually presents this relationship through a bar chart. The analysis aims to uncover trends indicating whether certain colors are more favorably rated than others, potentially guiding product development and marketing strategies.

# In[44]:


color_analysis = df.groupby('Color')['Rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=color_analysis.index, y=color_analysis.values)
plt.title('Impact of Color on Average Product Rating')
plt.xlabel('Color')
plt.ylabel('Average Rating')
plt.show()


# ## Size Impact Analysis
# To analyze the impact of product size on average ratings, this script groups data by 'Size' and calculates the mean rating for each size category. A bar plot visualizes these average ratings, highlighting how different sizes are perceived in terms of quality or appeal. This analysis is pivotal for understanding consumer preferences and optimizing product offerings for enhanced customer satisfaction.

# In[45]:


size_analysis = df.groupby('Size')['Rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=size_analysis.index, y=size_analysis.values)
plt.title('Impact of Size on Average Product Rating')
plt.xlabel('Size')
plt.ylabel('Average Rating')
plt.show()


# ## Brand Popularity and Rating Analysis
# This script segment evaluates the relationship between brand popularity and average product ratings. By grouping data by brand and calculating the mean rating and count of ratings, we can visualize how brand popularity correlates with consumer satisfaction. The scatter plot highlights patterns between the average rating and the number of ratings, offering insights into which brands are both popular and highly rated.

# In[46]:


brand_popularity_analysis = df.groupby('Brand')['Rating'].agg(['mean', 'count'])
brand_popularity_analysis.sort_values(by=['count', 'mean'], ascending=[False, False], inplace=True)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=brand_popularity_analysis, x='mean', y='count')
plt.title('Brand Popularity vs. Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Number of Ratings (Popularity)')
plt.show()


# ## Relationship Between Product Price and User Ratings
# The code snippet creates a scatter plot visualizing the relationship between product price and user ratings. The X-axis represents the product price, while the Y-axis displays user ratings. This visualization can help identify if there's a trend or pattern that suggests a correlation between the price of products and their ratings by users, which could be useful for assessing the impact on sales potential.

# In[49]:


# Creating the scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Price', y='Rating')
plt.title('Relationship Between Product Price and User Ratings')
plt.xlabel('Price')
plt.ylabel('User Rating')
plt.show()


# # Which product categories show the highest potential for sales increase through personalized recommendations?
#    The category analysis chart suggests that 'Kids' Fashion' has the highest potential for a sales increase as it has the highest average rating amongst the top categories, followed closely by 'Women's Fashion' and 'Men's Fashion'. Personalized recommendations in these categories could be especially effective.
# 
# # How does the relationship between product price and user ratings impact sales potential?
#   Based on the scatter plot, it appears there is a wide distribution of user ratings across different product prices. This suggests that price alone may not be a decisive factor in how users rate products, indicating that other factors might also play a significant role in user satisfaction and perceived value. Therefore, while price is an essential aspect, it should be considered alongside other product attributes when evaluating sales potential and customer satisfaction.
# # What are the key features of products that significantly influence their likelihood of being purchased when recommended?
#    The chart showing the impact of color and size on ratings indicate that certain colors and sizes receive higher ratings. Green and size 'S' seem to have the highest average ratings, implying that these attributes could influence a product’s likelihood to be purchased when recommended. Additionally, brand popularity is a significant feature, as shown by the scatterplot correlating the number of ratings (popularity) with the average rating.
# 
# 

# In[ ]:




