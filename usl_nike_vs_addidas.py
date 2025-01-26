
# importing necessary
# this will help in making the Python code more structured automatically (good coding practice)
#%load_ext nb_black

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# to scale the data using z-score
from sklearn.preprocessing import StandardScaler

# to compute distances
from scipy.spatial.distance import cdist

# to perform k-means clustering and compute silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# to visualize the elbow curve and silhouette scores
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# lets gives access to google drive
from google.colab import drive
drive.mount('/content/drive')

data=pd.read_csv('/content/drive/MyDrive/data_nike_vs_addidas_unsupervised.csv')
df=data.copy()

# checking top 5 rows
df.head(5)

# checking shape of data
df.shape

# checking data types
df.info()

"""- all data types are correct
- there are no missing values
"""

# checking descriptive statics
df.describe().T

# checking unique and missing values and duplicates
print(df.isna().sum().to_frame('missing values'))
print("there are",df.duplicated().sum(),"duplicates")
print(df.nunique().to_frame('unique values'))

"""## EDA

#### Product Name
"""

# checking unique product
df['Product Name'].unique()

# checking value of each product
df['Product Name'].value_counts()

"""#### Product ID"""

# checking product id unique values
df['Product ID'].nunique()

"""#### Listing Price"""

# checking listing price
df['Listing Price'].nunique()

# checking it with visual
plt.figure(figsize=(10,6))
sns.distplot(df['Listing Price'],color='y')
plt.show()

"""#### Sale Price"""

#  checking unique values in sale price
df['Sale Price'].nunique()

# checking saleprice with chart visual
plt.figure(figsize=(10,6))
sns.distplot(df['Sale Price'],color='r')
plt.show()

"""#### Discount"""

# checking unique values in discount
df['Discount'].unique()

# checking discount with histogram
plt.figure(figsize=(10,6))
sns.histplot(df['Discount'],color='g',kde='True')
plt.show()

"""#### Brand"""

# checking unique value in brand
df['Brand'].unique()

# checking all brand with visual chart
plt.figure(figsize=(10,6))
sns.countplot(df['Brand'],color='b')
plt.show()

"""#### Rating"""

# checking unique rating
df['Rating'].unique()

# checking rating with visual chart
plt.figure(figsize=(10,6))
sns.histplot(df['Rating'],color='m')
plt.show()

"""#### Reviews"""

# checking unique values in review
df['Reviews'].unique()

# checking it with visual
plt.figure(figsize=(10,6))
sns.distplot(df['Reviews'],color='g')
plt.show()

"""## Biovariate

#### Product Name vs Listing Price
"""

# checking product name vs listing price with visual chart
plt.figure(figsize=(200,10))
sns.barplot(data=df,x='Product Name',y='Listing Price',color='y')
plt.xticks(rotation=90)
plt.show()

"""#### Product Name vs Sale Price"""

# checking product name and sale price
plt.figure(figsize=(200,10))
sns.barplot(data=df,x='Product Name',y='Sale Price',color='r')
plt.xticks(rotation=90)
plt.show()

"""#### Product Name vs Discount"""

# lets check the product name vs discount on chart visual
plt.figure(figsize=(200,10))
sns.barplot(data=df,x='Product Name',y='Discount',color='g')
plt.xticks(rotation=90)
plt.show()

"""#### Brand vs Discount"""

# checking brand vs discount on visual chart
plt.figure(figsize=(10,6))
sns.barplot(data=df,x='Brand',y='Discount',color='b')
plt.show()

"""#### Brand vs Rating"""

# checking brand vs discount with visual chart
plt.figure(figsize=(10,6))
sns.barplot(data=df,x='Brand',y='Rating',color='m')
plt.show()

"""#### Brand vs Sale Price"""

# checking brand and sale price with visual chart
plt.figure(figsize=(10,6))
sns.barplot(data=df,x='Brand',y='Sale Price',color='r')
plt.show()

"""## Data Preprocessing"""

# Checking duplicates, missing values
print(df.isna().sum().to_frame('missing values'))
print("there are",df.duplicated().sum(),"duplicates")

"""- there are no missing values.
- there are 88 duplicates.
"""

# lets remove duplicates
df.drop_duplicates(inplace=True)

# checking outlier in dataset
df.boxplot(figsize=(10,6))
plt.show()

"""- there are few outlier present in sale_price and listing price
- assuming all data point is real.
"""

df.dtypes

# checking coreleation in dataset
num_col = ['Listing Price', 'Sale Price', 'Discount', 'Rating', 'Reviews']
plt.figure(figsize=(10,6))
sns.heatmap(df[num_col].corr(),annot=True)
plt.show()

# checking data distribution
sns.pairplot(df,diag_kind='kde')
plt.show()

# maping brand feature
df['Brand'] = df['Brand'].map({'Adidas CORE / NEO': 0, 'Adidas ORIGINALS': 1,'Nike':2,'Adidas SPORT PERFORMANCE':3,'Adidas Adidas ORIGINALS':4})

# Scaling the data set before clustering
scaler = StandardScaler()
subset = df[num_col].copy()
subset_scaled = scaler.fit_transform(subset)

# Creating a dataframe from the scaled data
subset_scaled_df = pd.DataFrame(subset_scaled, columns=subset.columns)

"""## K-MEANS CLUSTERING"""

k_means_df = subset_scaled_df.copy()

clusters = range(1, 15)
meanDistortions = []

for k in clusters:
    model = KMeans(n_clusters=k, random_state=1)
    model.fit(subset_scaled_df)
    prediction = model.predict(k_means_df)
    distortion = (
        sum(np.min(cdist(k_means_df, model.cluster_centers_, "euclidean"), axis=1))
        / k_means_df.shape[0]
    )

    meanDistortions.append(distortion)

    print("Number of Clusters:", k, "\tAverage Distortion:", distortion)

plt.plot(clusters, meanDistortions, "bx-")
plt.xlabel("k")
plt.ylabel("Average Distortion")
plt.title("Selecting k with the Elbow Method", fontsize=20)
plt.show()

model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(1, 15), timings=True)
visualizer.fit(k_means_df)  # fit the data to the visualizer
visualizer.show()  # finalize and render figure
plt.show()

"""**The appropriate value of k from the elbow curve seems to be 2 or 5.**

#### Let's check the silhouette scores
"""

sil_score = []
cluster_list = range(2, 15)
for n_clusters in cluster_list:
    clusterer = KMeans(n_clusters=n_clusters, random_state=1)
    preds = clusterer.fit_predict((subset_scaled_df))
    score = silhouette_score(k_means_df, preds)
    sil_score.append(score)
    print("For n_clusters = {}, the silhouette score is {})".format(n_clusters, score))

plt.plot(cluster_list, sil_score)
plt.show()

model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(2, 15), metric="silhouette", timings=True)
visualizer.fit(k_means_df)  # fit the data to the visualizer
visualizer.show()  # finalize and render figure
plt.show()

# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(2, random_state=1))
visualizer.fit(k_means_df)
visualizer.show()

# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(3, random_state=1))
visualizer.fit(k_means_df)
visualizer.show()

"""**Observations**

- For 4 clusters, there is a nick in the elbow plot and the silhouette score is high too.
- So, we will move ahead with k=2.

## Creating final model
"""

kmeans = KMeans(n_clusters=2, random_state=1)
kmeans.fit(k_means_df)

# creating a copy of the original data
df1 = df.copy()

# adding kmeans cluster labels to the original and scaled dataframes
k_means_df["KM_segments"] = kmeans.labels_
df1["KM_segments"] = kmeans.labels_

km_cluster_profile = df1.groupby("KM_segments").agg({
    'Listing Price': 'mean',
    'Sale Price': 'mean',
    'Discount': 'mean',
    'Rating': 'mean',
    'Reviews': 'mean',
    'Brand': 'mean'  # Or another appropriate aggregation for 'Brand'
})

km_cluster_profile["count_in_each_segment"] = (
    df1.groupby("KM_segments")["Brand"].count().values
)

km_cluster_profile.style.highlight_max(color="lightgreen", axis=0)

# let's see the names of the companies in each cluster
for cl in df1["KM_segments"].unique():
    print("In cluster {}, the following companies are present:".format(cl))
    print(df1[df1["KM_segments"] == cl]["Brand"].unique())
    print()

df1.groupby(["KM_segments", "Product Name"])['Brand'].count()

plt.figure(figsize=(20, 20))
plt.suptitle("Boxplot of numerical variables for each cluster")

# selecting numerical columns
num_col = df.select_dtypes(include=np.number).columns.tolist()

for i, variable in enumerate(num_col):
    plt.subplot(3, 4, i + 1)
    sns.boxplot(data=df1, x="KM_segments", y=variable)

plt.tight_layout(pad=2.0)

# Access the cluster labels assigned by KMeans
cluster_labels = k_means_df["KM_segments"]  # Or df1["KM_segments"]

# Create a new DataFrame or modify existing one to store the segmentation
segmented_data = df1.copy() #using original df1
#Now use the cluster labels directly, no need to recompute clusters
segmented_data['segment'] = cluster_labels

segmented_data=segmented_data.drop(['KM_segments'],axis=1)

# checking data frame with label
segmented_data
