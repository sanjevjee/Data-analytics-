**Data Description**

ID: Unique identifier for each individual.

Age: Age of the individual.

Gender: Gender (Male/Female).

Height: Height in cm.

Weight: Weight in kg.

BMI: Body Mass Index calculated from height and weight.

Label: Classification of obesity (e.g., Normal Weight, Overweight, Underweight, Obese)
"""

# importing necessary libraries
# to manupulate and transform data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# taking access of google drive
from google.colab import drive
drive.mount('/content/drive')

data= pd.read_csv('/content/drive/MyDrive/Obesity Classification.csv')
 df=data.copy()

df.head(5)

# checking shape of data
df.shape

"""- there are 108 rows and 7 columns."""

# checking datatypes
df.info()

"""- all data types is correct. and there are no missing values."""

# checking duplicates
df.duplicated().sum()

"""- there are no duplicates."""

# checking the unique values in every feature
df.nunique().to_frame('Unique Values').T

# checking discriptive statics of data
df.describe().T

"""## EDA

### Univariate analysis

#### ID
"""

# checking the no. of unique values
df['ID'].nunique()

"""- all id is unique values.

####  Age
"""

# checking the unique values
df['Age'].unique()

# check age distribution on chart
sns.histplot(df,x='Age',kde=True)

"""- age feature is right skewed.
- most of the people are from 20 to 60.

#### Height
"""

# checking the unique values in Height feature
df['Height'].unique()

# checking it with visual chart
sns.histplot(df,x='Height',kde=True);

"""#### Weight"""

# check unique values in Weight
df['Weight'].unique()

# check weight distribution with isual chart
sns.histplot(df,x='Weight',kde=True);

"""#### BMI"""

# checking unique values in BMI feature
df['BMI'].unique()

# checking bmi on chart
sns.histplot(df,x='BMI',kde=True);

"""####  Label"""

# checking unique values in Label feature
df['Label'].unique()

# checking label feature in chart feature
sns.countplot(df,x='Label',color="r");

"""#### Multivariate

#### age vs Lebels
"""

# checking as vs lebels in chart visual
pd.crosstab(df['Age'],df['Label']).plot(figsize=(10,5));

"""#### BMI vs Lebel"""

# checking bmi vs lebel
pd.crosstab(df['BMI'],df['Label']).plot(figsize=(10,5));

"""#### age vs weight"""

# checking age vs weight on chart visual
sns.scatterplot(df,x=df['Age'],y=df['Weight']);

"""#### age vs Bmi"""

# checking age vs Bmi on chart
sns.scatterplot(df,x=df['Age'],y=df['BMI']);

# copying data to make file safe
df1=df.copy()

"""## DATA PREPROCESSING"""

# checking missing values and duplicates in datasets
print(df1.isnull().sum())
print("there are",df1.duplicated().sum(),"Duplicates")

"""- there are o duplicates and 0 missing values."""

# checking oulier in data
plt.figure(figsize=(10,5))
sns.boxplot(df1)
plt.show()

"""- there are few outlier in age.assuming all data point is real."""

# checking correleation between features
num_col=df1.select_dtypes(include=np.number)
plt.figure(figsize=(10,5))
sns.heatmap(num_col.corr(),annot=True);

"""- Weight feature is strongle correleated with BMI.
- other feature are not strongly coreleated.
"""

# checking data distribution with pairplot
sns.pairplot(df1,);

# copying data
df2=df1.copy()

# lets make dependent variable label categorical to numeric
df2['Label']=df2['Label'].map({'Normal Weight':0,'Overweight':1,'Underweight':2,'Obese':3,})

# lets seprate independent and dependent variable
 X=df2.drop(['Label','ID'],axis=1)
 y=df2['Label']

# creating dummy variable for independent feature
X=pd.get_dummies(X,drop_first=True)
X=X.replace({True:1,False:0})

X.head()

# creating seprate set for train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd

def model_performance_classification_sklearn(model, predictors, target, average='macro'):
    """
    Function to compute different metrics to check classification model performance for multiclass problems.

    Parameters:
    model: classifier (fitted model)
    predictors: independent variables (features)
    target: dependent variable (true labels)
    average: str (default: 'macro') - The type of averaging to use for metrics.
             Use 'micro', 'macro', or 'weighted' for multiclass problems.

    Returns:
    DataFrame containing Accuracy, Recall, Precision, and F1-score.
    """
    # Predict using the independent variables
    pred = model.predict(predictors)

    # Compute metrics
    acc = accuracy_score(target, pred)  # Accuracy
    recall = recall_score(target, pred, average=average, zero_division=1)  # Recall
    precision = precision_score(target, pred, average=average, zero_division=1)  # Precision
    f1 = f1_score(target, pred, average=average, zero_division=1)  # F1-score

    # Create a DataFrame of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy": [acc],
            "Recall": [recall],
            "Precision": [precision],
            "F1": [f1],
        }
    )

    return df_perf

def confusion_matrix_sklearn(model, predictors, target, class_names=None):
    """
    To plot the confusion matrix with percentages for multiclass classification.

    Parameters:
    model: classifier (fitted model)
    predictors: independent variables (features)
    target: dependent variable (true labels)
    class_names: list (default: None) - Names of the classes for labeling the axes.
                 If None, numeric labels will be used.

    Returns:
    None (plots the confusion matrix)
    """
    # Predict the target values
    y_pred = model.predict(predictors)

    # Generate confusion matrix
    cm = confusion_matrix(target, y_pred)

    # Convert to percentages
    cm_percentage = cm.astype('float') / cm.sum() * 100

    # Format annotations (absolute values and percentages)
    labels = np.asarray(
        [
            f"{value:.0f}\n({percentage:.2f}%)"
            for value, percentage in zip(cm.flatten(), cm_percentage.flatten())
        ]
    ).reshape(cm.shape)

    # Use numeric class labels if class_names is not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

"""#### best model choosing with the help of k-fold cross validation

"""

# Import necessary libraries
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Scale features for algorithms sensitive to feature scaling (SVM, KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models including AdaBoost
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=101)),
    ('Decision Tree', DecisionTreeClassifier(random_state=101)),
    ('Random Forest', RandomForestClassifier(random_state=101)),
    ('SVM', SVC(random_state=101)),
    ('KNN', KNeighborsClassifier()),
    ('AdaBoost', AdaBoostClassifier(random_state=101))
]

# Define k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=101)

# Evaluate each model
results = []
for name, model in models:
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append((name, cv_results.mean(), cv_results.std()))

# Print results
print("Model Performance (Accuracy):")
for name, mean, std in results:
    print(f'{name}: Mean Accuracy = {mean:.4f}, Standard Deviation = {std:.4f}')

# Choose the best model based on mean accuracy
best_model_name, best_model_mean, _ = max(results, key=lambda item: item[1])
print(f'\nBest Model: {best_model_name} with Mean Accuracy = {best_model_mean:.4f}')

"""- best model for this dataset is decision tree and random forest.

#### Model tuning Decision Tree
"""

# Import necessary libraries
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV # add GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics  # Import the metrics module

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Define the parameter grid for Decision Tree
param_grid = {
    'criterion': ['gini', 'entropy'],  # Criteria for splitting
    'splitter': ['best', 'random'],   # Splitting strategy
    'max_depth': [None, 5, 10, 20, 50],  # Depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 5],    # Minimum samples at a leaf node
    'max_features': [None, 'sqrt', 'log2']  # Number of features to consider when looking for the best split
}

# Create a Decision Tree model
dt = DecisionTreeClassifier(random_state=101)

# Perform Grid Search
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid,
                           scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

# Fit Grid Search on training data
grid_search.fit(X_train, y_train)

# Display the best parameters and the best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Train a Decision Tree with the best parameters
best_dt = grid_search.best_estimator_

# Evaluate the best model on the test set
test_accuracy = best_dt.score(X_test, y_test)
print("Test Accuracy of Tuned Decision Tree:", test_accuracy)

"""#### Model Tuning Random forest"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees in the forest
    'criterion': ['gini', 'entropy'],     # Criteria for splitting
    'max_depth': [None, 10, 20, 30],      # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],      # Minimum samples to split an internal node
    'min_samples_leaf': [1, 2, 5],        # Minimum samples at a leaf node
    'max_features': ['sqrt', 'log2', None],  # Number of features to consider when looking for the best split
    'bootstrap': [True, False]            # Whether bootstrap samples are used when building trees
}

# Create a Random Forest model
rf = RandomForestClassifier(random_state=101)

# Perform Grid Search
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid,
                              scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

# Fit Grid Search on training data
grid_search_rf.fit(X_train, y_train)

# Display the best parameters and the best score
print("Best Parameters:", grid_search_rf.best_params_)
print("Best Cross-Validation Accuracy:", grid_search_rf.best_score_)

# Train a Random Forest with the best parameters
best_rf = grid_search_rf.best_estimator_

# Evaluate the best model on the test set
test_accuracy_rf = best_rf.score(X_test, y_test)
print("Test Accuracy of Tuned Random Forest:", test_accuracy_rf)

"""## MODEL BUILDING

#### Decision Tree
"""

# train decision tree
# Instantiate the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, max_features=None,
                                        min_samples_leaf=1, min_samples_split=2,
                                        splitter='random', random_state=101)

# Train the model
decision_tree.fit(X_train, y_train)

# checking model performance with train set
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
model_performance_classification_sklearn(decision_tree, X_train, y_train,)

# checking model performance with test set

model_performance_classification_sklearn(decision_tree, X_test, y_test,)

confusion_matrix_sklearn(decision_tree, X_train, y_train)

confusion_matrix_sklearn(decision_tree, X_test, y_test)

"""- decision tree is perfoming well in train set and not performing on test
- model is overfit.

#### Random Forest
"""

# Initialize the RandomForestClassifier with correct parameter syntax
rf = RandomForestClassifier(
    bootstrap=True,
    criterion='gini',
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)

# Train the model
rf.fit(X_train, y_train)

model_performance_classification_sklearn(rf, X_train, y_train,)

model_performance_classification_sklearn(rf, X_test, y_test,)

confusion_matrix_sklearn(rf, X_train, y_train)

confusion_matrix_sklearn(rf, X_test, y_test)

feature_names = X.columns
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

"""## Conclusion

- accuracy-97%
- recall-97%
- precision-97%
- f1_score-97%
- random forest is performing well on train and test set
- rf model is our best model.
- most important feature is weight followed by BMI,Height,age.

## Pipeline
"""

# pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Suppress warnings
warnings.filterwarnings('ignore')

# Mount Google Drive
drive.mount('/content/drive')

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/Obesity Classification.csv')
df = data.copy()

# Data preprocessing (assuming the same steps as in your original code)
df['Label'] = df['Label'].map({'Normal Weight': 0, 'Overweight': 1, 'Underweight': 2, 'Obese': 3})
X = df.drop('Label', axis=1)
y = df['Label']
X = pd.get_dummies(X, drop_first=True)
X = X.replace({True: 1, False: 0})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Create a pipeline with scaling and RandomForestClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scale the features
    ('rf', RandomForestClassifier(random_state=101))  # Use RandomForestClassifier as the estimator
])

# Define the parameter grid for the pipeline
param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__criterion': ['gini', 'entropy'],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 5],
    'rf__max_features': ['sqrt', 'log2', None],
    'rf__bootstrap': [True, False]
}

# Perform GridSearchCV with the pipeline
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)


# Evaluate the best model
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Use the best model for predictions (example)
y_pred = best_model.predict(X_test)

# Evaluate performance
print(model_performance_classification_sklearn(best_model, X_test, y_test))
confusion_matrix_sklearn(best_model, X_test, y_test)
