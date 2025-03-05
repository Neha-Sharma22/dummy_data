#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

num_rows = 10000

# Define the counts for Net_Units categories based on your provided distribution
# Original counts provided: 8400 (0 units), 450 (1-50), 256 (50-300), 342 (300-700), 112 (700-1500), 63 (>1500)
# These add up to 9623. To scale to 10,000, we use a proportional scaling factor.
scaling_factor = num_rows / 9623

cat0_count = int(round(8400 * scaling_factor))
cat1_count = int(round(450 * scaling_factor))
cat2_count = int(round(256 * scaling_factor))
cat3_count = int(round(342 * scaling_factor))
cat4_count = int(round(112 * scaling_factor))
cat5_count = int(round(63 * scaling_factor))

# Adjust to ensure total equals 10,000 (if necessary, tweak one category)
total_count = cat0_count + cat1_count + cat2_count + cat3_count + cat4_count + cat5_count
if total_count < num_rows:
    cat0_count += num_rows - total_count  # add the missing rows to category 0

print("Counts per category:", cat0_count, cat1_count, cat2_count, cat3_count, cat4_count, cat5_count)
print("Total rows:", cat0_count + cat1_count + cat2_count + cat3_count + cat4_count + cat5_count)

# Generate Net_Units for each category
net_units_cat0 = np.zeros(cat0_count, dtype=int)
net_units_cat1 = np.random.randint(1, 51, size=cat1_count)       # between 1 and 50
net_units_cat2 = np.random.randint(50, 301, size=cat2_count)     # between 50 and 300
net_units_cat3 = np.random.randint(300, 701, size=cat3_count)    # between 300 and 700
net_units_cat4 = np.random.randint(700, 1501, size=cat4_count)   # between 700 and 1500
net_units_cat5 = np.random.randint(1501, 3001, size=cat5_count)  # more than 1500 (here: 1501 to 3000)

# Concatenate and shuffle to randomize the order of sellers
net_units = np.concatenate([net_units_cat0, net_units_cat1, net_units_cat2, 
                            net_units_cat3, net_units_cat4, net_units_cat5])
np.random.shuffle(net_units)

# Create the DataFrame with the required columns
df = pd.DataFrame({
    'seller_id': ['seller_' + str(i) for i in range(1, num_rows + 1)],
    'Call_Attempt_Ratio': np.random.uniform(0, 1, num_rows),
    'Offer_Code_Rate': np.random.uniform(0, 1, num_rows),
    'Active_Leads_Ratio': np.random.uniform(0, 1, num_rows),
    'High_Potential_Leads': np.random.uniform(0, 1, num_rows),
    'Express_Delivery_Leads_Ratio': np.random.uniform(0, 1, num_rows),
    'Unique_Items_Ratio': np.random.uniform(0, 1, num_rows),
    'Add_to_Cart_Ratio': np.random.uniform(0, 1, num_rows),
    'High_Impression_Ratio': np.random.uniform(0, 1, num_rows),
    'Campaign_Engagement_Ratio': np.random.uniform(0, 1, num_rows),
    'Net_Units': net_units
})

# Simulate Gross_Units as Net_Units plus a random margin (ensuring Gross_Units > Net_Units)
gross_units = np.array([x + np.random.randint(1, 51) for x in net_units])
df['Unit_Efficiency'] = df['Net_Units'] / gross_units

# Optionally, if you do not need the Gross_Units column, you can leave it out
# df['Unit_Efficiency'] is our final computed metric


# In[23]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



# Feature selection (example)

features = ['Call_Attempt_Ratio', 'Offer_Code_Rate', 'Active_Leads_Ratio', 
            'High_Potential_Leads', 'Express_Delivery_Leads_Ratio', 
            'Unique_Items_Ratio', 'Add_to_Cart_Ratio', 'High_Impression_Ratio', 
            'Campaign_Engagement_Ratio']


bins = [0, 10, 50, 200, 500, float('inf')]
labels = [5, 4, 3, 2, 1]
df['Priority'] = pd.cut(df['Net_Units'], bins=bins, labels=labels, right=False)

# Prepare data
X = df[features]
y = df['Priority']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[39]:


# Scale numeric features  (Identify numeric columns first)
numeric_cols = X.select_dtypes(include=['number']).columns
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols]) #Fit and transform training data ONLY
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols]) #Use fitted scaler to transform test data


# In[43]:


#!pip install imblearn


# In[44]:


# Apply SMOTE for oversampling (do this *after* splitting and scaling)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# In[33]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define your model and parameter grid
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Create GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=5)

# Fit GridSearchCV to your training data
grid_search.fit(X_train_resampled, y_train_resampled)

# Now you can access the best estimator
best_model = grid_search.best_estimator_


# In[ ]:





# In[26]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate predictions
y_pred = best_model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['priority1', 'priority2', 'priority3', 'priority4', 'priority5'])
disp.plot()


# In[27]:


from sklearn.metrics import precision_score, recall_score

# Calculate precision and recall
precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for imbalanced datasets
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")


# In[28]:


from sklearn.metrics import f1_score

# Calculate F1-score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1}")


# In[29]:


from sklearn.metrics import roc_auc_score

# Predict probabilities
y_prob = best_model.predict_proba(X_test)

# Calculate ROC-AUC score
roc_auc = roc_auc_score(pd.get_dummies(y_test), y_prob, multi_class='ovr')
print(f"ROC-AUC Score: {roc_auc}")


# # predict the propensity

# In[34]:


# Predict priorities for all sellers using the same scaler
all_sellers_scaled = scaler.transform(X)  

# Scale all data using the same scaler
all_sellers_pred = best_model.predict(all_sellers_scaled)




# In[36]:



output_df = pd.DataFrame({
    'seller_id': df['seller_id'],
    'net_unit': df['Net_Units'],
    'priority': all_sellers_pred
})

# Map priority to string labels if needed
priority_map = {1: 'priority1', 2: 'priority2', 3: 'priority3', 4: 'priority4', 5: 'priority5'}
output_df['priority'] = output_df['priority'].map(priority_map)

print(output_df.head())


# In[37]:


output_df[["net_unit","priority"]].value_counts()


# In[17]:


df.columns


# In[ ]:




