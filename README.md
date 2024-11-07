#Design a data warehouse schema to identify dimensions, measures, and fact tables
#followed by implementing OLAP operations like roll-up, drill-down, slice, and dice on
#the data cube.

import numpy as np
import pandas as pd

product_data = {"Product_ID":[1,2,3,4],
               "Product_Name":["Home entertainment","Computer","Phone","Security"],
               "Category":["Electronics","Electronics","Electronics","Electronics"],
               "Brand":["BrandA","BrandB","BrandC","BrandD"] }

product_df = pd.DataFrame(product_data)

time_data = {"Time_ID":[1,2,3,4],
            "Date":["01-01-2024","14-01-2024","22-02-2024","16-03-2024"],
            "Month":["January","January","February","March"],
            "Quarter":["Q1","Q1","Q1","Q1"],
            "Year":["2024","2024","2024","2024"]}

time_df = pd.DataFrame(time_data)

store_data = {"Store_ID":[1,2,3,4],
             "Store_Location":["Vancouver","Toronto","NewYork","Chicago"],
             "Store_Name":["StoreA","StoreA","StoreB","StoreB"],
             "Country":["Canada","Canada","USA","USA"]}

store_df = pd.DataFrame(store_data)

sales_data = {"Sales_ID":[1,2,3,4],
             "Product_ID":[1,2,3,4],
             "Time_ID":[1,2,3,4],
             "Store_ID":[1,2,3,4],
             "Sales":[10,20,30,40],
             "Revenue":[1000,2000,3000,4000]}

sales_df = pd.DataFrame(sales_data)

print("Product Dimension:\n",product_df)
print("Time Dimension:\n",time_df)
print("Store Dimension:\n",store_df)
print("Sales Dimension:\n",sales_df)


#roll up
rollup_df = sales_df.merge(store_df,on="Store_ID").groupby(["Country","Store_Name"]).agg({"Sales":'sum',"Revenue":'sum'})
print("Roll up results:\n",rollup_df)

#slice
slice_df = sales_df.merge(product_df,on="Product_ID").query("Product_Name == 'Home entertainment'")
print("slice operation result:\n",slice_df)


#dice
dice_df = sales_df.merge(time_df,on= "Time_ID").merge(store_df,on="Store_ID").query("Quarter == 'Q1' and Country in ['Canada'] and Store_Name  == 'StoreA'")
print("dice operation result:\n",dice_df)

2)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Create the DataFrame
data = {
    'ID': [1, 2, 3, 4, 5],
    'Age': [25, 30, 22, 45, 35],
    'Income': [50000, 60000, 55000, 80000, 70000],
    'Height': [170, 180, 160, 175, 165],
    'Gender': ['M', 'F', 'M', 'F', 'M'],
    'Purchase Amount': [200, 250, 150, 300, 220]
}

df = pd.DataFrame(data)

# Step 1: Data Cleaning
# Check for missing values
print("Missing values before cleaning:")
print(df.isnull().sum())

# For simplicity, let's assume there are no missing values and move to duplicates
# Check for duplicates
print("\nDuplicates before cleaning:")
print(df.duplicated().sum())

# Remove duplicates if any (none in this case)
df = df.drop_duplicates()

# Step 2: Data Normalization

# Separate features and target
features = df[['Age', 'Income', 'Height', 'Purchase Amount']]
target = df['Gender']

# Normalize numerical features using Min-Max Scaling
scaler = MinMaxScaler()
normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# Encode categorical features (Gender) using Label Encoding
label_encoder = LabelEncoder()
encoded_target = label_encoder.fit_transform(target)

# Combine normalized features with encoded target
processed_df = normalized_features.copy()
processed_df['Gender'] = encoded_target

print("\nNormalized and encoded data:")
print(processed_df)

# Step 3: Dimensionality Reduction

# Apply PCA to reduce the number of features
pca = PCA(n_components=2)  # Reduce to 2 components
pca_result = pca.fit_transform(normalized_features)

3)
from itertools import combinations
from collections import defaultdict

def generate_candidates(itemsets, length):
    """Generate candidate itemsets of a given length from frequent itemsets."""
    candidates = set()
    itemsets_list = list(itemsets)
    for i in range(len(itemsets_list)):
        for j in range(i + 1, len(itemsets_list)):
            l1 = list(itemsets_list[i])[:length-2]
            l2 = list(itemsets_list[j])[:length-2]
            if l1 == l2:
                candidate = itemsets_list[i].union(itemsets_list[j])
                if len(candidate) == length:
                    candidates.add(candidate)
    return candidates

def apriori(transactions, min_support):
    """Apriori algorithm to find frequent itemsets."""
    # Step 1: Generate frequent itemsets of length 1
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1
   
    # Prune infrequent itemsets
    freq_itemsets = {itemset for itemset, count in item_counts.items() if count >= min_support}
    all_freq_itemsets = dict.fromkeys(freq_itemsets, 0)
    for itemset in freq_itemsets:
        all_freq_itemsets[itemset] = item_counts[itemset]

    k = 2
    while freq_itemsets:
        # Generate candidate itemsets of length k
        candidates = generate_candidates(freq_itemsets, k)
        item_counts.clear()
        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    item_counts[candidate] += 1
       
        # Prune infrequent itemsets
        freq_itemsets = {itemset for itemset, count in item_counts.items() if count >= min_support}
        for itemset in freq_itemsets:
            all_freq_itemsets[itemset] = item_counts[itemset]
        k += 1
   
    return all_freq_itemsets

# Example usage
transactions = [
    {'I1', 'I2', 'I5'},
    {'I2', 'I4'},
    {'I2', 'I3'},
    {'I1', 'I2','I4'},
    {'I1', 'I3'},
    {'I2','I3'},
    {'I1','I3'},
    {'I1','I2','I3','I5'},
    {'I1','I2','I3'}
]

min_support = 2
frequent_itemsets = apriori(transactions, min_support)
print("Frequent Itemsets:")
for itemset, count in frequent_itemsets.items():
    print(f"{set(itemset)}: {count}")



# Create a DataFrame with PCA results
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

print("\nPCA result (Dimensionality Reduced Data):")
print(pca_df)

4)
import pandas as pd

import matplotlib.pyplot as plt

import networkx as nx

from collections import defaultdict

from itertools import combinations


class FPNode:

    def __init__(self, name, count, parent=None):  # Fixed constructor name

        self.name, self.count, self.parent = name, count, parent

        self.children, self.next_node = {}, None


class FPTree:

    def __init__(self, min_support):  # Fixed constructor name

        self.min_support = min_support

        self.root = FPNode('root', 1)

        self.header_table = defaultdict(list)


    def add_transaction(self, transaction):

        node = self.root

        for item in transaction:

            node = node.children.setdefault(item, FPNode(item, 0, node))

            node.count += 1

            self.header_table[item].append(node)


    def build_tree(self, transactions):

        item_counts = defaultdict(int)

        for transaction in transactions:

            for item in set(transaction):

                item_counts[item] += 1

        sorted_items = [item for item in sorted(item_counts, key=item_counts.get, reverse=True) if item_counts[item] >= self.min_support]

        for transaction in transactions:

            filtered_transaction = [item for item in sorted_items if item in transaction]

            self.add_transaction(filtered_transaction)


    def mine_patterns(self, prefix, min_support):

        patterns = []

        for item, nodes in self.header_table.items():

            support = sum(node.count for node in nodes)

            if support >= min_support:

                new_prefix = prefix + [item]

                patterns.append((new_prefix, support))

                patterns.extend(self.build_conditional_tree(nodes).mine_patterns(new_prefix, min_support))

        return patterns


    def build_conditional_tree(self, nodes):

        conditional_tree = FPTree(self.min_support)

        for node in nodes:

            path = []

            while node.parent and node.parent.name != 'root':

                path.append(node.parent.name)

                node = node.parent

            for _ in range(node.count):

                conditional_tree.add_transaction(reversed(path))

        return conditional_tree


    def draw_tree(self):

        G = nx.DiGraph()

        self._add_edges(self.root, G)

        pos = nx.spring_layout(G, seed=42)  # Calculate positions after edges are added

        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray")

        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'))

        plt.show()



    def _add_edges(self, node, G):

        for child in node.children.values():

            G.add_edge(node.name, child.name, label=str(child.count))

            self._add_edges(child, G)


def compute_confidence(patterns, min_support):

    rule_confidences = []

    for pattern, support in patterns:

        for subset in combinations(pattern, len(pattern) - 1):

            subset = list(subset)

            subset_support = next((s for p, s in patterns if sorted(p) == sorted(subset)), 0)

            if subset_support:

                confidence = support / subset_support

                if confidence >= min_support:

                    rule_confidences.append((sorted(subset), sorted(set(pattern) - set(subset)), confidence))

    return rule_confidences


def find_most_frequent_pattern(patterns):

    return max(patterns, key=lambda x: x[1], default=None)


def main():

    transactions = [

        ['E', 'K', 'M', 'N', 'O', 'Y'],

        ['D', 'E', 'K', 'N', 'O', 'Y'],

        ['A', 'E', 'K', 'M'],

        ['C', 'K', 'M', 'U', 'Y'],

        ['C', 'E', 'I', 'K', 'O', 'O']

    ]

    min_support_ratio, min_confidence = 0.6, 0.6

    min_support = int(min_support_ratio * len(transactions))


    fp_tree = FPTree(min_support)

    fp_tree.build_tree(transactions)

    fp_tree.draw_tree()


    frequent_patterns = fp_tree.mine_patterns([], min_support)

    most_frequent_pattern = find_most_frequent_pattern(frequent_patterns)


    print("\nFrequent Itemsets:")

    for pattern, support in frequent_patterns:

        print(f"Itemset: {set(pattern)}, Support: {support / len(transactions):.4f}")


    if most_frequent_pattern:

        print(f"\nMost Frequent Pattern: {set(most_frequent_pattern[0])}, Support: {most_frequent_pattern[1] / len(transactions):.4f}")


    print("\nAssociation Rules:")

    rule_confidences = compute_confidence(frequent_patterns, min_confidence)

    for antecedent, consequent, confidence in rule_confidences:

        if most_frequent_pattern and set(antecedent).issubset(set(most_frequent_pattern[0])):

            print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence:.4f}")


if __name__ == "__main__":

    main()

5)
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: 0))
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        class_counts = pd.Series(y).value_counts()
        total_samples = len(y)

        # Calculate prior probabilities
        self.class_probs = {cls: count / total_samples for cls, count in class_counts.items()}

        # Calculate likelihoods
        feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        for i in range(len(y)):
            label = y[i]
            features = X[i]
            for j, feature in enumerate(features):
                feature_counts[label][j][feature] += 1

        for cls in self.classes:
            feature_count = len(feature_counts[cls])
            for feature_index, feature_dict in feature_counts[cls].items():
                total_count = sum(feature_dict.values())
                for feature_value, count in feature_dict.items():
                    self.feature_probs[cls][(feature_index, feature_value)] = count / total_count

    def predict(self, X):
        predictions = []
        for features in X:
            class_scores = {}
            for cls in self.classes:
                score = np.log(self.class_probs[cls])
                for i, feature in enumerate(features):
                    score += np.log(self.feature_probs[cls].get((i, feature), 1e-6))  # Use smoothing for unseen features
                class_scores[cls] = score
            predictions.append(max(class_scores, key=class_scores.get))
        return predictions
# Example dataset
data = pd.DataFrame({
    'Feature1': [1, 2, 1, 2, 1],
    'Feature2': [1, 1, 2, 2, 2],
    'Class': ['A', 'A', 'B', 'B', 'B']
})

X = data[['Feature1', 'Feature2']].values
y = data['Class'].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Initialize and train the classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Predictions:", y_pred)

6)
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Create dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [18, 17, 19, 16, 22],
    'Country': ['India', 'India', 'USA', 'UK', 'India'],
    'Can_vote': ['Yes', 'No', 'Yes', 'No', 'Yes']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical variables
le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])

# Prepare features and target
X = df[['Age', 'Country']]  # Include Age and Country as features
y = df['Can_vote']  # Target: Can_vote (Yes/No)

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier(max_depth=1)  # Simple tree with depth 1
clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=['Age', 'Country'], class_names=le_target.classes_, filled=True, rounded=True, impurity=False)
plt.title('Decision Tree: Voting Eligibility Based on Age and Country')
plt.show()

# Function to get user input and predict voting eligibility
def predict_voting_eligibility(age, country):
    # Encode the country input
    country_encoded = le_country.transform([country])[0]
    # Create DataFrame for prediction
    X_pred = pd.DataFrame({'Age': [age], 'Country': [country_encoded]})
    prediction = clf.predict(X_pred)
    return le_target.inverse_transform(prediction)[0]

# Example usage: get age and country from user and predict
user_age = int(input("Enter age: "))
user_country = input("Enter country: ")
result = predict_voting_eligibility(user_age, user_country)
print(f"Can vote: {result}")

7)
import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X, k):
    """Randomly initialize centroids from the data points."""
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    """Assign each data point to the nearest centroid."""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    """Update centroids to be the mean of points in each cluster."""
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def k_means(X, k, max_iters=100, tolerance=1e-4):
    """Perform K-Means clustering."""
    centroids = initialize_centroids(X, k)
    print("Initial centroids:\n", centroids)
   
    for iteration in range(max_iters):
        # Step 1: Assign clusters
        labels = assign_clusters(X, centroids)
       
        # Print results for this iteration
        print(f"\nIteration {iteration + 1}:")
        print("Labels:\n", labels)
        print("Centroids:\n", centroids)
       
        # Step 2: Update centroids
        new_centroids = update_centroids(X, labels, k)
       
        # Check for convergence
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tolerance):
            print(f"\nConverged after {iteration + 1} iterations.")
            break
       
        centroids = new_centroids
   
    return centroids, labels

def get_user_input():
    """Get data points and number of clusters from the user."""
    print("Enter the number of clusters (k):")
    k = int(input())
   
    print("Enter the number of data points:")
    num_points = int(input())
   
    data_points = []
    for i in range(num_points):
        print(f"Enter coordinates for data point {i+1} (comma-separated):")
        point = list(map(float, input().strip().split(',')))
        data_points.append(point)
   
    return np.array(data_points), k

def plot_clusters(X, labels, centroids):
    """Plot the data points and centroids."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering')
    plt.legend()
    plt.show()

def main():
    X, k = get_user_input()
   
    if X.shape[1] != 2:
        print("This implementation only supports 2D data for visualization. Proceeding with dimensionality reduction.")
        X = X[:, :2]
   
    centroids, labels = k_means(X, k)
   
    print("Final Centroids:\n", centroids)
    print("Final Labels:\n", labels)
   
    plot_clusters(X, labels, centroids)

if __name__ == "__main__":
    main()

8)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#DBSCAN_dataset = pd.read_csv('Mall_Customers.csv')  
data_dict = [
    {'CustomerID': 1, 'Gender': 'Male', 'Age': 19, 'Annual Income (k$)': 15, 'Spending Score (1-100)': 39},
    {'CustomerID': 2, 'Gender': 'Male', 'Age': 21, 'Annual Income (k$)': 15, 'Spending Score (1-100)': 81},
    {'CustomerID': 3, 'Gender': 'Female', 'Age': 20, 'Annual Income (k$)': 16, 'Spending Score (1-100)': 6},
    {'CustomerID': 4, 'Gender': 'Female', 'Age': 23, 'Annual Income (k$)': 16, 'Spending Score (1-100)': 77},
    {'CustomerID': 5, 'Gender': 'Female', 'Age': 31, 'Annual Income (k$)': 17, 'Spending Score (1-100)': 40},
    {'CustomerID': 6, 'Gender': 'Female', 'Age': 22, 'Annual Income (k$)': 17, 'Spending Score (1-100)': 76},
    {'CustomerID': 7, 'Gender': 'Female', 'Age': 35, 'Annual Income (k$)': 18, 'Spending Score (1-100)': 6},
    {'CustomerID': 8, 'Gender': 'Female', 'Age': 23, 'Annual Income (k$)': 18, 'Spending Score (1-100)': 94},
    {'CustomerID': 9, 'Gender': 'Male', 'Age': 64, 'Annual Income (k$)': 19, 'Spending Score (1-100)': 3},
    {'CustomerID': 10, 'Gender': 'Female', 'Age': 30, 'Annual Income (k$)': 19, 'Spending Score (1-100)': 72},
    {'CustomerID': 11, 'Gender': 'Male', 'Age': 67, 'Annual Income (k$)': 19, 'Spending Score (1-100)': 14},
    {'CustomerID': 12, 'Gender': 'Female', 'Age': 35, 'Annual Income (k$)': 19, 'Spending Score (1-100)': 99},
    {'CustomerID': 13, 'Gender': 'Female', 'Age': 58, 'Annual Income (k$)': 20, 'Spending Score (1-100)': 15},
    {'CustomerID': 14, 'Gender': 'Female', 'Age': 24, 'Annual Income (k$)': 20, 'Spending Score (1-100)': 77},
    {'CustomerID': 15, 'Gender': 'Male', 'Age': 37, 'Annual Income (k$)': 20, 'Spending Score (1-100)': 13},
    {'CustomerID': 16, 'Gender': 'Male', 'Age': 22, 'Annual Income (k$)': 20, 'Spending Score (1-100)': 79},
    {'CustomerID': 17, 'Gender': 'Female', 'Age': 35, 'Annual Income (k$)': 21, 'Spending Score (1-100)': 35},
    {'CustomerID': 18, 'Gender': 'Male', 'Age': 20, 'Annual Income (k$)': 21, 'Spending Score (1-100)': 66},
    {'CustomerID': 19, 'Gender': 'Male', 'Age': 52, 'Annual Income (k$)': 23, 'Spending Score (1-100)': 29},
    {'CustomerID': 20, 'Gender': 'Female', 'Age': 35, 'Annual Income (k$)': 23, 'Spending Score (1-100)': 98}
]

# Convert the list of dictionaries into a DataFrame
DBSCAN_dataset = pd.DataFrame(data_dict)

def manual_clustering(data, eps=10, min_points=5):
    clusters = [0] * len(data)  
    cluster_id = 0

    def distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def expand_cluster(point_idx, neighbors):
        clusters[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if clusters[neighbor_idx] == 0:
                clusters[neighbor_idx] = cluster_id
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= min_points:
                    neighbors.extend(new_neighbors)
            i += 1

    def region_query(point_idx):
        point = data.iloc[point_idx].values
        return [i for i in range(len(data)) if distance(point, data.iloc[i].values) <= eps]

    for i in range(len(data)):
        if clusters[i] == 0:  
            neighbors = region_query(i)
            if len(neighbors) >= min_points:  
                cluster_id += 1
                expand_cluster(i, neighbors)
            else:
                clusters[i] = -1  

    return clusters

DBSCAN_dataset['Cluster'] = manual_clustering(DBSCAN_dataset[['Annual Income (k$)', 'Spending Score (1-100)']], eps=10, min_points=5)

unique_clusters = DBSCAN_dataset['Cluster'].unique()
if len(unique_clusters) < 6:  
    cluster_mapping = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(unique_clusters, start=1)}
    DBSCAN_dataset['Cluster'] = DBSCAN_dataset['Cluster'].map(cluster_mapping)

outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster'] == 1]

fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                data=DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1],
                hue='Cluster', ax=axes[0], palette='Set2', legend='full', s=200)

sns.scatterplot(x='Age', y='Spending Score (1-100)',
                data=DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1],
                hue='Cluster', palette='Set2', ax=axes[1], legend='full', s=200)

# Update: Increase the size of the outliers and plot using sns.scatterplot for consistency
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                data=outliers, s=100, color="k", label="Outliers", ax=axes[0])

sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                data=outliers, s=100, color="k", label="Outliers", ax=axes[1])

axes[0].legend(title="Clusters + Outliers")

plt.tight_layout()
plt.show()

9)
 import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Generate some synthetic data
np.random.seed(42)
# Inliers: Normally distributed data
X_inliers = 0.5 * np.random.normal(size=(200, 2))
# Outliers: Uniformly distributed data
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
# Combine the data
X = np.concatenate([X_inliers, X_outliers], axis=0)

# Convert to DataFrame for better visualization
data = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the Local Outlier Factor model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# Predict outliers (-1 for outliers, 1 for inliers)
y_pred = lof.fit_predict(X_scaled)
# Compute the negative outlier factor (the lower, the more abnormal)
outlier_scores = -lof.negative_outlier_factor_

# Add predictions and scores to the DataFrame
data['Outlier'] = y_pred
data['Outlier Score'] = outlier_scores

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(data['Feature 1'], data['Feature 2'],
            c=y_pred, cmap='coolwarm', s=100, edgecolor='k', label='Data points')
plt.colorbar(label='Outlier Label (1: Inlier, -1: Outlier)')
plt.title("Local Outlier Factor (LOF) Outlier Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Highlight the outliers with red circles
for i in range(len(data)):
    if y_pred[i] == -1:
        plt.scatter(data.iloc[i, 0], data.iloc[i, 1], facecolor='none', s=150, linewidths=2)

plt.legend(["Inliers", "Outliers"])
plt.show()

# Display outliers
print("Outliers detected:")
print(data[data['Outlier'] == -1])

10)
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

def create_sample_graphs():
    # Sample data: Creating 3 graphs
    G1 = nx.Graph()
    G1.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])
   
    G2 = nx.Graph()
    G2.add_edges_from([(1, 2), (2, 3), (3, 5), (5, 6)])
   
    G3 = nx.Graph()
    G3.add_edges_from([(1, 2), (2, 4), (4, 5), (5, 6)])
   
    return [G1, G2, G3]

def get_subgraphs(graph, size):
    """ Generate all connected subgraphs of a given size from the input graph. """
    subgraphs = []
    nodes = list(graph.nodes())
   
    for combo in combinations(nodes, size):
        subgraph = graph.subgraph(combo)
        if nx.is_connected(subgraph):
            subgraphs.append(subgraph)
   
    return subgraphs

def find_frequent_subgraphs(graphs, subgraph_size, min_support):
    """ Find frequent subgraphs across a list of graphs. """
    subgraph_count = {}
   
    for graph in graphs:
        subgraphs = get_subgraphs(graph, subgraph_size)
       
        for subgraph in subgraphs:
            key = frozenset(subgraph.edges())
            if key in subgraph_count:
                subgraph_count[key] += 1
            else:
                subgraph_count[key] = 1
   
    frequent_subgraphs = {key: count for key, count in subgraph_count.items() if count >= min_support}
   
    return frequent_subgraphs

def draw_graph(graph, title):
    """ Draw and display the graph. """
    plt.figure()
    pos = nx.spring_layout(graph)  # positions for all nodes
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=700)
    plt.title(title)
    plt.show()

# Creating sample graphs
graphs = create_sample_graphs()

# Finding frequent subgraphs of size 2 with a minimum support of 2
frequent_subgraphs = find_frequent_subgraphs(graphs, subgraph_size=2, min_support=2)

# Displaying the results
print("Frequent Subgraphs:")
for subgraph_edges, count in frequent_subgraphs.items():
    print(f"Subgraph edges: {set(subgraph_edges)}, Count: {count}")

# Draw original graphs
for i, graph in enumerate(graphs, start=1):
    draw_graph(graph, f"Original Graph {i}")

# Draw frequent subgraphs
for subgraph_edges in frequent_subgraphs.keys():
    edges = list(subgraph_edges)
    subgraph = nx.Graph()
    subgraph.add_edges_from(edges)
    draw_graph(subgraph, "Frequent Subgraph")

11)
import networkx as nx
import numpy as np

def calculate_reachability(graph, node, epsilon):
    neighbors = list(graph.neighbors(node))
    reachability = []
    for neighbor in neighbors:
        if len(set(graph.neighbors(neighbor)).intersection(neighbors)) >= epsilon:
            reachability.append(neighbor)
    return reachability

def scan(graph, epsilon, min_pts):
    visited = set()
    clusters = []
    outliers = []

    for node in graph.nodes():
        if node in visited:
            continue
        visited.add(node)
        
        reachability = calculate_reachability(graph, node, epsilon)
        
        if len(reachability) < min_pts:
            outliers.append(node)
            continue
        
        # Start a new cluster
        new_cluster = [node]
        to_expand = reachability[:]
        
        while to_expand:
            current = to_expand.pop()
            if current in visited:
                continue
            visited.add(current)
            new_cluster.append(current)

            current_reachability = calculate_reachability(graph, current, epsilon)
            for neighbor in current_reachability:
                if neighbor not in visited:
                    to_expand.append(neighbor)

        clusters.append(new_cluster)
    
    return clusters, outliers

# Example usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.Graph()
    edges = [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4), (7, 8)]
    G.add_edges_from(edges)

    # Parameters for SCAN
    epsilon = 2  # Minimum number of shared neighbors to consider a node connected
    min_pts = 1  # Minimum number of points to form a cluster

    clusters, outliers = scan(G, epsilon, min_pts)

    print("Clusters:", clusters)
    print("Outliers:", outliers)

12)
import itertools
from collections import defaultdict

class Graph:
    def __init__(self):
        self.edges = defaultdict(list)
        self.nodes = set()
    
    def add_edge(self, u, v, label):
        self.edges[u].append((v, label))
        self.edges[v].append((u, label))
        self.nodes.update([u, v])

class gSpan:
    def __init__(self, min_support=2):
        self.min_support = min_support
        self.frequent_subgraphs = []
        self.graph_database = []
    
    def add_graph(self, graph):
        self.graph_database.append(graph)
    
    def get_support(self, subgraph):
        """Check support of a subgraph pattern within the graph database."""
        count = 0
        for graph in self.graph_database:
            if self.is_subgraph(graph, subgraph):
                count += 1
        return count
    
    def is_subgraph(self, graph, subgraph):
        """Check if subgraph is part of a larger graph."""
        # This is a simplified check and needs a more rigorous implementation
        # to perform subgraph isomorphism.
        return set(subgraph.nodes).issubset(set(graph.nodes))
    
    def search(self):
        """Generate initial patterns and begin DFS exploration."""
        initial_patterns = self.generate_initial_patterns()
        for pattern in initial_patterns:
            self.dfs(pattern)
    
    def generate_initial_patterns(self):
        """Generate 1-edge subgraphs as starting patterns."""
        patterns = []
        for graph in self.graph_database:
            for u in graph.nodes:
                for v, label in graph.edges[u]:
                    if u < v:  # To avoid duplicating edges
                        new_pattern = Graph()
                        new_pattern.add_edge(u, v, label)
                        patterns.append(new_pattern)
        return patterns
    
    def dfs(self, subgraph):
        """Depth-first search to grow the subgraph."""
        support = self.get_support(subgraph)
        if support < self.min_support:
            return
        self.frequent_subgraphs.append(subgraph)

        extensions = self.generate_extensions(subgraph)
        for extension in extensions:
            self.dfs(extension)
    
    def generate_extensions(self, subgraph):
        """Generate candidate extensions for the given subgraph."""
        extensions = []
        for node in subgraph.nodes:
            for graph in self.graph_database:
                for neighbor, label in graph.edges[node]:
                    if neighbor not in subgraph.nodes:
                        new_subgraph = Graph()
                        new_subgraph.edges = subgraph.edges.copy()
                        new_subgraph.nodes = subgraph.nodes.copy()
                        new_subgraph.add_edge(node, neighbor, label)
                        extensions.append(new_subgraph)
        return extensions

# Example usage
if __name__ == "__main__":
    # Define a sample graph database
    g1 = Graph()
    g1.add_edge(1, 2, 'A')
    g1.add_edge(2, 3, 'B')
    g1.add_edge(3, 4, 'C')

    g2 = Graph()
    g2.add_edge(1, 2, 'A')
    g2.add_edge(2, 3, 'B')
    g2.add_edge(3, 5, 'D')

    g3 = Graph()
    g3.add_edge(1, 3, 'A')
    g3.add_edge(3, 4, 'B')
    g3.add_edge(4, 5, 'C')

    # Run gSpan on the graph database
    gspan = gSpan(min_support=2)
    gspan.add_graph(g1)
    gspan.add_graph(g2)
    gspan.add_graph(g3)
    
    gspan.search()

    print("Frequent subgraphs found:")
    for i, subgraph in enumerate(gspan.frequent_subgraphs, 1):
        print(f"Subgraph {i}:")
        for node, edges in subgraph.edges.items():
            for (neighbor, label) in edges:
                print(f"{node} --{label}--> {neighbor}")
        print()
