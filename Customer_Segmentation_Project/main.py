import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("customers.csv")

# Encode category
le = LabelEncoder()
data['ProductCategory'] = le.fit_transform(data['ProductCategory'])

# Select features
X = data[['TotalPurchases', 'TotalSpent']]

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Segment'] = kmeans.fit_predict(X)

# Recommendation logic
def recommend(segment):
    if segment == 2:
        return "Premium Electronics, Smart Gadgets"
    elif segment == 1:
        return "Grocery Offers, Daily Essentials"
    else:
        return "Budget Fashion, Discount Items"

data['Recommendation'] = data['Segment'].apply(recommend)

# Output
print("\nCustomer Segmentation & Recommendations:\n")
print(data[['CustomerID', 'Segment', 'Recommendation']])

# Visualization
plt.scatter(data['TotalPurchases'], data['TotalSpent'], c=data['Segment'])
plt.xlabel("Total Purchases")
plt.ylabel("Total Spent")
plt.title("Customer Segmentation")
plt.show()
