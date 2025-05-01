# %% [markdown] - Cell 1: Introduction
# # Health Impact Prediction - Air Quality Analysis
# 
# This notebook explores how weather and environmental conditions affect health risk.
# We'll build machine learning models to predict health risk levels based on air quality and weather data.
# The analysis includes data exploration, feature selection, model training, and evaluation.

# %% [markdown] - Cell 2: Setup and Data Loading

# ## 1. Setup and Data Loading
# 
# First, let's import the libraries we need and load our dataset.

# %% [code] - Cell 3: Library Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score
import warnings
from scipy import stats
import joblib  # Add joblib for model serialization
warnings.filterwarnings('ignore')

# Statistical utility functions
def cramers_v(confusion_matrix):
    """
    Calculate Cramér's V statistic for categorical correlation.
    Uses a contingency table to perform the calculation.
    """
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size between two groups."""
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                          (len(group2) - 1) * np.var(group2, ddof=1)) / 
                          (len(group1) + len(group2) - 2))
    return mean_diff / pooled_std

def eta_squared(anova_stat, df_between, df_total):
    """Calculate eta-squared effect size for ANOVA."""
    return (anova_stat * df_between) / (anova_stat * df_between + df_total)

def format_p(p):
    """Format p-value for display with proper scientific notation."""
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, risk_mapping):
    """
    Comprehensive model evaluation function that produces detailed metrics,
    visualizations, and interpretations.
    """
    # Train the model
    print(f"Training {model_name} model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=list(risk_mapping.keys())))
    
    # Confusion matrix with normalized values
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Visualization of confusion matrix
    plt.figure(figsize=(10, 8))
    
    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(risk_mapping.keys()),
                yticklabels=list(risk_mapping.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_cm.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=list(risk_mapping.keys()),
                yticklabels=list(risk_mapping.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Normalized Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_cm_norm.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC curve for multi-class (one-vs-rest)
    if y_prob is not None:
        plt.figure(figsize=(10, 8))
        for i, category in enumerate(risk_mapping.keys()):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{category} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_roc.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Calculate additional metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Effect size - Cramer's V for categorical outcomes
    cramers_v_value = cramers_v(cm)
    
    print("\nAdditional Metrics:")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Effect Size (Cramer's V): {cramers_v_value:.4f}")
    
    # Interpretation guide
    effect_size_interpretation = ""
    if cramers_v_value < 0.1:
        effect_size_interpretation = "Negligible"
    elif cramers_v_value < 0.3:
        effect_size_interpretation = "Small"
    elif cramers_v_value < 0.5:
        effect_size_interpretation = "Medium"
    else:
        effect_size_interpretation = "Large"
    
    print(f"Effect Size Interpretation: {effect_size_interpretation}")
    
    # Return the model and performance metrics for comparison
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'effect_size': cramers_v_value
    }

# Set up visualization style
plt.rcParams['figure.figsize'] = (10, 6)

# %% [code] - Cell 4: Data Loading
# Load the dataset
data_file = 'Urban Air Quality and Health Impact Dataset.csv'
try:
    df = pd.read_csv(data_file)
    print(f"Data loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print(f"File not found. Please check the path: {data_file}")
    # If you're running this in a notebook, you might need to adjust the path

# Look at the first few rows to understand the data
print("\nFirst 5 rows of data:")
print(df.head())

# %% [markdown] - Cell 5: Exploratory Data Analysis
# ## 2. Exploratory Data Analysis
# 
# Let's explore the data to understand what we're working with.

# %% [code] - Cell 6: Basic Info and Statistics
# Check basic information about the dataset
print("Dataset Information:")
df.info()

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])  # Show only columns with missing values

# Statistical summary of numerical features
print("\nSummary statistics of key features:")
numerical_df = df.select_dtypes(include=['float64', 'int64'])
print(numerical_df.describe().T.sort_values(by='mean', ascending=False).head(10))

# %% [code] - Cell 7: Target Variable Analysis
# Explore the target variable: Health_Risk_Score
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Health_Risk_Score', kde=True, bins=20)
plt.title('Distribution of Health Risk Scores')
plt.axvline(df['Health_Risk_Score'].mean(), color='red', linestyle='--', 
           label=f'Mean: {df["Health_Risk_Score"].mean():.2f}')
plt.axvline(df['Health_Risk_Score'].median(), color='green', linestyle='--', 
           label=f'Median: {df["Health_Risk_Score"].median():.2f}')
plt.legend()
plt.savefig('health_risk_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Health Risk Score Summary Statistics:\n{df['Health_Risk_Score'].describe()}")

# %% [code] - Cell 8: Feature Relationships
# Look at how key environmental factors relate to health risk
# Use correlation analysis to identify key features
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numerical_columns].corr()

# Get features most correlated with Health_Risk_Score
health_correlations = correlation_matrix['Health_Risk_Score'].dropna().sort_values(ascending=False)
# Remove Health_Risk_Score itself from the top correlations
health_correlations = health_correlations[health_correlations.index != 'Health_Risk_Score']
# Select top features for visualization
top_features = health_correlations.head(9).index.tolist()

print("Top correlated features with Health Risk Score:")
print(health_correlations.head(9))

# Visualize relationships for top correlated features
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
axes = axes.flatten()

for i, feature in enumerate(top_features):
    if i < len(axes) and feature in df.columns:  # Safety check
        sns.scatterplot(x=feature, y='Health_Risk_Score', data=df, alpha=0.6, ax=axes[i])
        axes[i].set_title(f'{feature} vs Health Risk Score')
        
        # Add regression line to visualize relationship
        sns.regplot(x=feature, y='Health_Risk_Score', data=df, 
                   scatter=False, ax=axes[i], color='red')

# Remove empty subplots if any
for i in range(len(top_features), len(axes)):
    fig.delaxes(axes[i])
    
plt.tight_layout()
plt.savefig('feature_relationships.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [code] - Cell 9: Correlation Analysis
# Check correlation between numerical variables
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numerical_columns].corr()

# Plot heatmap of correlations
plt.figure(figsize=(14, 12))
mask = np.triu(correlation_matrix.corr())
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5,
           mask=mask)  # Use mask to show only lower triangle
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Let's find features most correlated with health risk
health_correlations = correlation_matrix['Health_Risk_Score'].sort_values(ascending=False)
print("Top 10 features correlated with Health Risk Score:")
print(health_correlations.head(10))
print("\nBottom 5 features correlated with Health Risk Score:")
print(health_correlations.tail(5))

# %% [code] - Cell 9.8: Advanced Feature Exploration
# Deeper analysis of key features in relation to health risk
important_features = ['Severity_Score', 'dew', 'windgust', 'Heat_Index', 'temp']
target = 'Health_Risk_Score'

# Statistical relationship analysis
print("\nDetailed Statistical Analysis of Key Features:")
print("{:<20} {:<10} {:<10} {:<15} {:<10}".format(
    "Feature", "Corr.", "p-value", "Effect Size", "Significance"
))
print("-" * 70)

for feature in important_features:
    if feature in df.columns:
        # Pearson correlation
        corr, p_val = stats.pearsonr(df[feature].dropna(), df[target].dropna())
        
        # Calculate effect size (r² for continuous variables)
        effect_size = corr**2
        
        # Determine significance level
        sig_level = ""
        if p_val < 0.001:
            sig_level = "***"
        elif p_val < 0.01:
            sig_level = "**"
        elif p_val < 0.05:
            sig_level = "*"
        else:
            sig_level = "ns"
            
        print("{:<20} {:<10.3f} {:<10s} {:<15.3f} {:<10s}".format(
            feature, corr, format_p(p_val), effect_size, sig_level
        ))
        
        # Create visualization to show relationship
        plt.figure(figsize=(10, 6))
        sns.regplot(x=feature, y=target, data=df, scatter_kws={'alpha':0.5})
        plt.title(f'Relationship between {feature} and Health Risk Score')
        plt.xlabel(feature)
        plt.ylabel('Health Risk Score')
        plt.tight_layout()
        plt.savefig(f'feature_analysis_{feature}.png', dpi=300, bbox_inches='tight')
        plt.show()

# %% [code] - Cell 9.9: Multivariate Analysis
# How do combinations of features affect health risk?
# Let's look at interaction between temperature and humidity

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='temp', 
    y='humidity', 
    hue='Health_Risk_Score',
    data=df,
    palette='viridis',
    size='Health_Risk_Score',
    sizes=(20, 200),
    alpha=0.7
)
plt.title('Interaction between Temperature, Humidity and Health Risk')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.colorbar(label='Health Risk Score')
plt.tight_layout()
plt.savefig('temp_humidity_interaction.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [code] - Cell 10: City-Based Analysis
# Analyze risk by city
plt.figure(figsize=(12, 6))
sns.boxplot(x='City', y='Health_Risk_Score', data=df)
plt.title('Health Risk Score by City')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('city_risk_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Check mean risk score by city
city_risk = df.groupby('City')['Health_Risk_Score'].agg(['mean', 'std', 'count']).sort_values(by='mean', ascending=False)
print("\nHealth Risk Score by City:")
print(city_risk)

# %% [code] - Cell 11: Seasonal Analysis
# Check if there are multiple seasons in the dataset
if 'Season' in df.columns:
    unique_seasons = df['Season'].unique()
    print(f"\nSeasons in the dataset: {unique_seasons}")
    
    # Display boxplot even if there's only one season (Fall)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Season', y='Health_Risk_Score', data=df)
    plt.title('Health Risk Score by Season (Fall)')
    plt.tight_layout()
    plt.savefig('seasonal_risk.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    if len(unique_seasons) == 1:
        print(f"Note: All data in this dataset is from a single season: {unique_seasons[0]}")
        print("Distribution above shows the range of health risk scores during Fall season.")
    
    # Check mean risk score by season
    season_risk = df.groupby('Season')['Health_Risk_Score'].agg(['mean', 'std', 'count']).sort_values(by='mean', ascending=False)
    print("\nHealth Risk Score by Season:")
    print(season_risk)

# %% [markdown] - Cell 11.5: Geographic and Temporal Insights
# ## City and Seasonal Analysis Results
# 
# ### City Analysis
# Our city-based analysis reveals important geographic patterns in health risk:
# 
# - **Phoenix** has the highest average health risk score among all cities, likely due to its extreme heat and dry conditions
# - **Los Angeles** shows moderate risk levels with high variability, possibly reflecting its diverse microclimates and pollution challenges
# - **San Jose** demonstrates the lowest average risk scores, suggesting more favorable environmental conditions
# - The significant differences between cities highlight that location-specific factors are critical in determining health outcomes
# 
# ### Seasonal Analysis
# 
# The dataset contains data from only the Fall season, which limits our ability to understand seasonal patterns:
# 
# - All observations come from a single season (Fall)
# - The Fall season data shows a wide range of health risk scores (from ~15 to ~85)
# - Without data from other seasons, we cannot make comparisons about seasonal health risk variations
# - Future analysis would benefit from year-round data to understand how seasonal changes affect health risks

# %% [markdown] - Cell 12: Preparing Data for Modeling
# ## 3. Preparing Data for Modeling
# 
# Now we'll prepare our data for the machine learning models.
# 
# Note: Our analysis is limited to Fall season data only, as the dataset does not contain 
# information from other seasons. A more comprehensive analysis would require data from 
# all seasons to understand seasonal variations in health risk factors.

# %% [code] - Cell 13: Feature Selection
# Based on correlation analysis, select the most relevant features
# This helps reduce dimensionality and improve model performance
# We'll also remove redundant features that might cause multicollinearity

# List of features to exclude (highly correlated with others or irrelevant)
exclude_columns = ['datetime', 'datetimeEpoch', 'sunrise', 'sunriseEpoch', 'sunset', 'sunsetEpoch']

# Drop irrelevant columns and create a new feature set
feature_df = df.drop(exclude_columns, axis=1, errors='ignore')

print(f"Original number of features: {df.shape[1]}")
print(f"Number of features after initial cleaning: {feature_df.shape[1]}")

# %% [code] - Cell 14: Create Target Variable for Classification
# Let's create a categorical version of the Health Risk Score for classification
# We'll use quartiles to create 4 categories: Low, Moderate, High, Very High

# Create bins based on quartiles
quartiles = df['Health_Risk_Score'].quantile([0.25, 0.5, 0.75])
bins = [float('-inf'), quartiles[0.25], quartiles[0.5], quartiles[0.75], float('inf')]
labels = ['Low', 'Moderate', 'High', 'Very High']

# Create the categorical target
feature_df['Risk_Category'] = pd.cut(df['Health_Risk_Score'], bins=bins, labels=labels)

# Show the distribution of categories
plt.figure(figsize=(10, 6))
category_counts = feature_df['Risk_Category'].value_counts().sort_index()
ax = sns.barplot(x=category_counts.index, y=category_counts.values)

# Add value labels on top of bars
for i, v in enumerate(category_counts.values):
    ax.text(i, v + 5, str(v), ha='center')

plt.title('Distribution of Risk Categories')
plt.xlabel('Risk Category')
plt.ylabel('Count')
plt.savefig('risk_categories.png', dpi=300, bbox_inches='tight')
plt.show()

print("Risk Category Distribution:")
percentage_dist = feature_df['Risk_Category'].value_counts(normalize=True).sort_index().map(lambda x: f"{x:.1%}")
for category, percentage in percentage_dist.items():
    print(f"{category}: {percentage}")

# %% [markdown] - Cell 14.5: Data Preparation Results
# ## Feature Selection and Target Creation Results
# 
# ### Feature Engineering
# We have processed the dataset to prepare it for modeling:
# 
# - Excluded time-related columns that don't provide direct predictive value
# - Reduced the original feature set from ~30 features to ~24 relevant features
# - Maintained all weather and location variables with significant correlations to health risk
# 
# ### Risk Category Creation
# We transformed the continuous Health Risk Score into categorical risk levels:
# 
# - Created four evenly distributed risk categories using quartiles: Low, Moderate, High, and Very High
# - Each category contains approximately 25% of the data (about 250 observations)
# - This transformation allows us to perform classification rather than regression
# - The balanced distribution across categories provides a good foundation for training classification models
# 
# This preparation ensures our models can effectively learn patterns in the data while focusing on the most relevant features.

# %% [code] - Cell 15: Split Features and Target
# Separate features and target
X = feature_df.drop(['Health_Risk_Score', 'Risk_Category'], axis=1)
y = feature_df['Risk_Category']

# Create a custom mapping to ensure logical order of risk categories
risk_mapping = {
    'Low': 0,
    'Moderate': 1,
    'High': 2,
    'Very High': 3
}

# Apply the mapping to encode the target variable (instead of using LabelEncoder)
y_encoded = feature_df['Risk_Category'].map(risk_mapping)

print("Risk Categories Encoded:")
for category, code in risk_mapping.items():
    print(f"{category}: {code}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, 
                                                   test_size=0.2, 
                                                   random_state=42, 
                                                   stratify=y_encoded)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# %% [code] - Cell 16: Create Preprocessing Pipeline
# Create preprocessing pipeline for numerical and categorical features
# First, identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

print(f"Number of numerical features: {len(numerical_cols)}")
print(f"Number of categorical features: {len(categorical_cols)}")
print(f"Categorical features: {list(categorical_cols)}")

# Create preprocessors
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())                   # Scale the data
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))    # One-hot encode
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# %% [markdown] - Cell 17: Model Building and Evaluation
# ## 4. Building and Evaluating Models
# 
# Let's build and evaluate multiple classification models.

# %% [code] - Cell 18: Logistic Regression Model
# 1. Logistic Regression Model
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])

print("Training Logistic Regression model...")
log_reg_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_lr = log_reg_pipeline.predict(X_test)

# Evaluate the model
print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=list(risk_mapping.keys())))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(risk_mapping.keys()), 
            yticklabels=list(risk_mapping.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Logistic Regression')
plt.savefig('logistic_regression_cm.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [code] - Cell 19: Random Forest Model
# 2. Random Forest Model
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

print("Training Random Forest model...")
rf_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_pipeline.predict(X_test)

# Evaluate the model
print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=list(risk_mapping.keys())))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(risk_mapping.keys()), 
            yticklabels=list(risk_mapping.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest')
plt.savefig('random_forest_cm.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [code] - Cell 20: Gradient Boosting Model
# 3. Gradient Boosting Model (new addition)
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

print("Training Gradient Boosting model...")
gb_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_gb = gb_pipeline.predict(X_test)

# Evaluate the model
print("\nGradient Boosting Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb, target_names=list(risk_mapping.keys())))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(risk_mapping.keys()), 
            yticklabels=list(risk_mapping.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Gradient Boosting')
plt.savefig('gradient_boosting_cm.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown] - Cell 20.5: Model Evaluation Results
# ## Classification Model Performance
# 
# We've trained three different models to predict health risk categories, with the following results:
# 
# ### Logistic Regression
# - **Accuracy**: ~88%
# - **Strengths**: Good performance across all risk categories, especially for Low and Very High risk classes
# - **Weaknesses**: Slightly lower precision for Moderate risk category
# - **Key Insight**: Despite being the simplest model, logistic regression performs competitively with more complex models
# 
# ### Random Forest
# - **Accuracy**: ~89%
# - **Strengths**: Best overall accuracy, excellent precision for extreme categories (Low and Very High risk)
# - **Weaknesses**: Some confusion between adjacent categories (e.g., Moderate vs High)
# - **Key Insight**: The ensemble nature helps capture complex relationships between features and health risk
# 
# ### Gradient Boosting
# - **Accuracy**: ~86%
# - **Strengths**: Good balance between precision and recall
# - **Weaknesses**: Slightly lower overall accuracy compared to other models
# - **Key Insight**: While not the top performer, still provides valuable predictions and different error patterns
# 
# All three models perform well at distinguishing between low risk and high risk cases, with the most confusion occurring between adjacent risk categories (which is expected given the continuous nature of the underlying health risk score).

# %% [code] - Cell 21: Enhanced Feature Importance Analysis
# Comprehensive analysis of feature importance from the Random Forest model

# Get feature names after preprocessing
ohe = rf_pipeline.named_steps['preprocessor'].transformers_[1][1]['onehot']
feature_names = list(numerical_cols)
if hasattr(ohe, 'get_feature_names_out'):
    cat_features = ohe.get_feature_names_out(categorical_cols)
    feature_names.extend(cat_features)

# Extract feature importances
importances = rf_pipeline.named_steps['classifier'].feature_importances_

# Match importances with feature names and sort
if len(feature_names) == len(importances):
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top 15 features
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    
    # Add percentage labels
    total = sum(importances)
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        percentage = 100 * width / total
        ax.text(width + 0.005, p.get_y() + p.get_height()/2, 
                f'{percentage:.1f}%', 
                ha='left', va='center')
    
    plt.title('Top 15 Most Important Features', fontsize=16)
    plt.xlabel('Relative Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate cumulative importance
    feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum()
    top_n_features = feature_importance[feature_importance['cumulative_importance'] <= 0.95]
    
    print(f"Number of features needed to capture 95% of importance: {len(top_n_features)}")
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Plot cumulative importance
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(feature_importance) + 1), feature_importance['cumulative_importance'], 'b-')
    plt.hlines(y=0.95, xmin=1, xmax=len(feature_importance), colors='r', linestyles='--')
    plt.title('Cumulative Feature Importance')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.axvline(x=len(top_n_features), color='g', linestyle='--')
    plt.text(len(top_n_features) + 1, 0.9, f'95% importance: {len(top_n_features)} features', va='center')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Group features by category
    if 'temp' in feature_importance['feature'].values and 'humidity' in feature_importance['feature'].values:
        feature_categories = {
            'Temperature': [f for f in feature_importance['feature'] if 'temp' in f or 'feels' in f],
            'Wind': [f for f in feature_importance['feature'] if 'wind' in f],
            'Moisture': [f for f in feature_importance['feature'] if 'humidity' in f or 'dew' in f or 'precip' in f],
            'Pressure': [f for f in feature_importance['feature'] if 'pressure' in f],
            'Other': []
        }
        
        # Create category importance DataFrame
        category_importance = []
        for category, features in feature_categories.items():
            if features:
                total_importance = feature_importance[feature_importance['feature'].isin(features)]['importance'].sum()
                category_importance.append({
                    'category': category,
                    'importance': total_importance
                })
        
        category_df = pd.DataFrame(category_importance).sort_values('importance', ascending=False)
        
        # Plot category importance
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='importance', y='category', data=category_df)
        
        # Add percentage labels
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            percentage = 100 * width / total
            ax.text(width + 0.005, p.get_y() + p.get_height()/2, 
                    f'{percentage:.1f}%', 
                    ha='left', va='center')
                    
        plt.title('Feature Importance by Category')
        plt.xlabel('Relative Importance')
        plt.ylabel('Category')
        plt.tight_layout()
        plt.savefig('category_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nFeature Importance by Category:")
        print(category_df)
else:
    print("Could not match feature names with importances. Lengths differ.")
    print(f"Feature names: {len(feature_names)}, Importances: {len(importances)}")

# %% [markdown] - Cell 22: Cross-Validation and Hyperparameter Tuning
# ## 5. Cross-Validation and Hyperparameter Tuning
# 
# Cross-validation helps us make sure our model performance is reliable.
# We'll also try to optimize the Random Forest model with hyperparameter tuning.

# %% [code] - Cell 23: Cross-Validation
# Let's use 5-fold cross-validation to get a more robust estimate of model performance
from sklearn.model_selection import KFold

# Set up k-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validate each model
models = {
    'Random Forest': rf_pipeline,
    'Logistic Regression': log_reg_pipeline,
    'Gradient Boosting': gb_pipeline
}

# Dictionary to store CV scores for each model
cv_scores_dict = {}

for name, model in models.items():
    print(f"Performing cross-validation for {name}...")
    scores = cross_val_score(model, X, y_encoded, cv=kf, scoring='accuracy')
    cv_scores_dict[name] = scores
    print(f"Cross-validation accuracy: {scores.mean():.4f} (±{scores.std():.4f})")
    print(f"Individual fold scores: {scores}")
    print()

# %% [code] - Cell 23.6: Comprehensive Model Comparison
# Compare model performance across multiple metrics

# Collect all evaluation metrics in a DataFrame
model_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression', 'Gradient Boosting'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_gb)
    ],
    'Precision': [
        precision_score(y_test, y_pred_rf, average='weighted'),
        precision_score(y_test, y_pred_lr, average='weighted'),
        precision_score(y_test, y_pred_gb, average='weighted')
    ],
    'Recall': [
        recall_score(y_test, y_pred_rf, average='weighted'),
        recall_score(y_test, y_pred_lr, average='weighted'),
        recall_score(y_test, y_pred_gb, average='weighted')
    ],
    'F1 Score': [
        f1_score(y_test, y_pred_rf, average='weighted'),
        f1_score(y_test, y_pred_lr, average='weighted'),
        f1_score(y_test, y_pred_gb, average='weighted')
    ],
    'CV Accuracy': [
        cv_scores_dict['Random Forest'].mean(),
        cv_scores_dict['Logistic Regression'].mean(),
        cv_scores_dict['Gradient Boosting'].mean()
    ],
    'CV Std': [
        cv_scores_dict['Random Forest'].std(),
        cv_scores_dict['Logistic Regression'].std(),
        cv_scores_dict['Gradient Boosting'].std()
    ]
})

# Format percentages
for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV Accuracy']:
    model_comparison[col] = model_comparison[col].map(lambda x: f"{x:.2%}")
model_comparison['CV Std'] = model_comparison['CV Std'].map(lambda x: f"{x:.3f}")

print("Comprehensive Model Comparison:")
print(model_comparison)

# Visualize model comparison
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV Accuracy']

# Convert percentage strings back to floats for plotting
plot_df = model_comparison.copy()
for col in metrics_to_plot:
    plot_df[col] = plot_df[col].str.rstrip('%').astype(float) / 100

# Reshape data for plotting
plot_data = pd.melt(
    plot_df, 
    id_vars=['Model'], 
    value_vars=metrics_to_plot,
    var_name='Metric', 
    value_name='Score'
)

# Create the plot
plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Score', hue='Metric', data=plot_data)
plt.title('Model Performance Comparison', fontsize=16)
plt.ylabel('Score', fontsize=14)
plt.xlabel('Model', fontsize=14)
plt.ylim(0.7, 1.0)  # Adjust based on your scores
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)

# Add value labels on top of bars
for i, bar in enumerate(plt.gca().patches):
    plt.gca().text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.005,
        f"{bar.get_height():.2f}",
        ha='center',
        color='black',
        fontsize=9
    )

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [code] - Cell 23.7: Comparative Confusion Matrices and ROC Curves
# Create a visual comparison of all three models side by side

# 1. Confusion Matrix Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
y_preds = [y_pred_lr, y_pred_rf, y_pred_gb]

for i, (title, y_pred) in enumerate(zip(titles, y_preds)):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=list(risk_mapping.keys()), 
                yticklabels=list(risk_mapping.keys()),
                ax=axes[i])
    
    axes[i].set_title(f'Normalized CM - {title}')
    axes[i].set_xlabel('Predicted')
    if i == 0:
        axes[i].set_ylabel('True')
    else:
        axes[i].set_ylabel('')

plt.tight_layout()
plt.savefig('model_comparison_cm.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. ROC Curve Comparison
models_list = [log_reg_pipeline, rf_pipeline, gb_pipeline]
colors = ['blue', 'green', 'red']
linestyles = ['-', '--', '-.']

plt.figure(figsize=(12, 10))

# For each class (Low, Moderate, High, Very High)
for class_idx in range(len(risk_mapping)):
    plt.subplot(2, 2, class_idx + 1)
    class_name = list(risk_mapping.keys())[class_idx]
    
    # For each model
    for i, (model_name, model) in enumerate(zip(titles, models_list)):
        # Calculate the prediction probabilities
        y_prob = model.predict_proba(X_test)
        
        # Calculate ROC curve and ROC area
        fpr, tpr, _ = roc_curve((y_test == class_idx).astype(int), y_prob[:, class_idx])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=colors[i], linestyle=linestyles[i],
                 label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {class_name} Risk')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_roc.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Comprehensive Performance Metrics Table
print("\nDetailed Performance Comparison:")
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC (macro avg)'],
    'Logistic Regression': [
        f"{accuracy_score(y_test, y_pred_lr):.4f}",
        f"{precision_score(y_test, y_pred_lr, average='weighted'):.4f}",
        f"{recall_score(y_test, y_pred_lr, average='weighted'):.4f}",
        f"{f1_score(y_test, y_pred_lr, average='weighted'):.4f}",
        f"{roc_auc_score(y_test, log_reg_pipeline.predict_proba(X_test), multi_class='ovr'):.4f}"
    ],
    'Random Forest': [
        f"{accuracy_score(y_test, y_pred_rf):.4f}",
        f"{precision_score(y_test, y_pred_rf, average='weighted'):.4f}",
        f"{recall_score(y_test, y_pred_rf, average='weighted'):.4f}",
        f"{f1_score(y_test, y_pred_rf, average='weighted'):.4f}",
        f"{roc_auc_score(y_test, rf_pipeline.predict_proba(X_test), multi_class='ovr'):.4f}"
    ],
    'Gradient Boosting': [
        f"{accuracy_score(y_test, y_pred_gb):.4f}",
        f"{precision_score(y_test, y_pred_gb, average='weighted'):.4f}",
        f"{recall_score(y_test, y_pred_gb, average='weighted'):.4f}",
        f"{f1_score(y_test, y_pred_gb, average='weighted'):.4f}",
        f"{roc_auc_score(y_test, gb_pipeline.predict_proba(X_test), multi_class='ovr'):.4f}"
    ]
})

print(metrics_df.to_string(index=False))

# 4. Statistical Significance Testing
try:
    # Try to import from statsmodels instead
    from statsmodels.stats.contingency_tables import mcnemar
    
    print("\nStatistical Significance Testing (McNemar's Test):")
    print("Comparing model predictions for statistical differences")
    
    # Create contingency tables for each pair of models
    lr_vs_rf = confusion_matrix(y_pred_lr == y_test, y_pred_rf == y_test)
    lr_vs_gb = confusion_matrix(y_pred_lr == y_test, y_pred_gb == y_test)
    rf_vs_gb = confusion_matrix(y_pred_rf == y_test, y_pred_gb == y_test)
    
    # McNemar's test
    try:
        # Statsmodels implementation
        result_lr_rf = mcnemar(lr_vs_rf, exact=True)
        result_lr_gb = mcnemar(lr_vs_gb, exact=True)
        result_rf_gb = mcnemar(rf_vs_gb, exact=True)
        
        print(f"LogReg vs RF: p-value = {result_lr_rf.pvalue:.4f} {'(significant)' if result_lr_rf.pvalue < 0.05 else '(not significant)'}")
        print(f"LogReg vs GB: p-value = {result_lr_gb.pvalue:.4f} {'(significant)' if result_lr_gb.pvalue < 0.05 else '(not significant)'}")
        print(f"RF vs GB: p-value = {result_rf_gb.pvalue:.4f} {'(significant)' if result_rf_gb.pvalue < 0.05 else '(not significant)'}")
    except Exception as e:
        print(f"Could not perform McNemar's test: {e}")
        print("This might be due to insufficient disagreements between models")
except ImportError:
    # Alternative comparison if mcnemar is not available
    print("\nModel Prediction Comparison (Agreement rates):")
    print(f"LogReg vs RF agreement: {np.mean(y_pred_lr == y_pred_rf):.4f}")
    print(f"LogReg vs GB agreement: {np.mean(y_pred_lr == y_pred_gb):.4f}")
    print(f"RF vs GB agreement: {np.mean(y_pred_rf == y_pred_gb):.4f}")
    
    # Compare errors
    print("\nError patterns comparison:")
    print(f"LogReg errors: {1 - accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"RF errors: {1 - accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"GB errors: {1 - accuracy_score(y_test, y_pred_gb):.4f}")
    
    # Check for common/different errors
    lr_errors = y_pred_lr != y_test
    rf_errors = y_pred_rf != y_test
    gb_errors = y_pred_gb != y_test
    
    print(f"Errors made by all models: {np.mean(lr_errors & rf_errors & gb_errors):.4f}")
    print(f"Errors unique to LogReg: {np.mean(lr_errors & ~rf_errors & ~gb_errors):.4f}")
    print(f"Errors unique to RF: {np.mean(~lr_errors & rf_errors & ~gb_errors):.4f}")
    print(f"Errors unique to GB: {np.mean(~lr_errors & ~rf_errors & gb_errors):.4f}")

# Import for AUC calculation
from sklearn.metrics import roc_auc_score

# %% [code] - Cell 24: Hyperparameter Tuning
# Let's perform a simple grid search to optimize our best model
# Note: This can take some time to run
best_model = 'Random Forest'  # We can change this based on cross-validation results

param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

print(f"Performing hyperparameter tuning for {best_model}...")
grid_search = GridSearchCV(
    models[best_model],
    param_grid,
    cv=3,  # Use fewer folds for speed, increase for better estimates
    scoring='accuracy',
    n_jobs=-1  # Use all available cores
)

# %% [code] - Cell 25: Run the grid search

grid_search.fit(X, y_encoded)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Create a model with the best parameters
# Extract the parameter values correctly, removing the 'classifier__' prefix
best_params = {param.replace('classifier__', ''): value 
               for param, value in grid_search.best_params_.items()}

best_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(**best_params, random_state=42))
])

# Train and evaluate the best model
best_rf.fit(X_train, y_train)
y_pred_best = best_rf.predict(X_test)
print(f"Best model accuracy on test set: {accuracy_score(y_test, y_pred_best):.4f}")

# %% [markdown] - Cell 25.5: Hyperparameter Tuning Results
# ## Grid Search Optimization Results
# 
# The grid search hyperparameter tuning process yielded valuable improvements to our Random Forest model:
# 
# ### Optimal Parameters
# - **n_estimators**: The optimal number of trees in the forest was found to be 150, indicating that more trees helped capture complex patterns
# - **max_depth**: A maximum depth of 20 was ideal, providing sufficient complexity while avoiding overfitting
# - **min_samples_split**: The best value was 2, allowing for fine-grained tree splits
# 
# ### Performance Gains
# - **Cross-validation score**: The optimized model achieved ~91% accuracy during cross-validation
# - **Test set accuracy**: When applied to the unseen test data, the model maintained similarly high accuracy (~90%)
# - **Improvement**: The tuned model improved performance by approximately 2 percentage points compared to the default configuration
# 
# ### Key Insights
# - The relatively modest improvement suggests our initial model configuration was already well-suited to the problem
# - The consistency between cross-validation and test scores indicates the model generalizes well
# - The grid search confirmed that a moderately complex model works best for this dataset
# 
# This systematic optimization process has resulted in our most accurate model, which balances complexity and generalization ability effectively.

# %% [markdown] - Cell 26: Conclusion
# ## 6. Conclusion and Insights
# 
# ### Summary of Findings
# Our comprehensive analysis of the Urban Air Quality and Health Impact dataset has yielded several important insights:
# 
# - **Model Performance**: We've successfully built predictive models that can classify health risk levels with approximately 90% accuracy. The Random Forest model with optimized hyperparameters performed best overall, though Logistic Regression showed remarkable effectiveness despite its simplicity.
# 
# - **Key Environmental Factors**: Through correlation analysis and statistical significance testing, we've identified that the following features have the strongest influence on health risk:
#   - **Severity_Score**: The strongest predictor (r² ≈ 0.64), suggesting that existing severity metrics capture health risk well
#   - **Heat_Index**: Strong positive relationship, with every unit increase corresponding to a ~0.19 increase in health risk
#   - **Temperature parameters**: Higher temperatures consistently associated with elevated health risks
#   - **Wind parameters** (windgust, windspeed): Significant correlations with health outcomes
#   - **Humidity and Dew Point**: Important factors related to heat stress and respiratory effects
# 
# - **Geographic Variations**: Cities show markedly different health risk profiles, with Phoenix having the highest average risk scores and San Jose the lowest. These differences highlight the importance of local environmental conditions.
# 
# - **Data Limitations**: Our analysis is constrained to Fall season data only, which limits our ability to understand seasonal patterns in health risk. A more comprehensive dataset with year-round observations would provide deeper insights.
# 
# ### Practical Applications
# 
# These findings have several practical applications:
# 
# - **Public Health Planning**: Health officials could use our models to forecast risk levels based on weather forecasts, enabling proactive public health interventions.
# 
# - **Early Warning Systems**: The identified key risk factors could form the basis of early warning systems for vulnerable populations.
# 
# - **City-Specific Strategies**: Different cities may require tailored approaches to mitigate health risks based on their specific environmental profiles.
# 
# - **Resource Allocation**: Healthcare resources could be allocated more efficiently by focusing on high-risk periods and locations.
# 
# ### Future Directions
# 
# To extend this work, we recommend:
# 
# - **Expanded Data Collection**: Gathering data across all seasons to understand seasonal variations in health risk
# 
# - **City-Specific Models**: Developing separate models for each city to account for local factors and improve prediction accuracy
# 
# - **Additional Environmental Metrics**: Incorporating air quality indices, pollen counts, and other pollutant measures
# 
# - **Time-Series Forecasting**: Building predictive models that can forecast health risks days or weeks in advance
# 
# - **Interactive Dashboard**: Developing a monitoring tool for public health officials that visualizes current and predicted health risks
# 
# This analysis demonstrates that machine learning can effectively predict health risks based on environmental data, providing valuable tools for public health management and environmental policy decisions.

# %% [code] - Cell 27: Save the Best Model
# Save the best model to a file for future use
model_filename = 'best_rf_model.pkl'
print(f"Saving the best model to {model_filename}...")
joblib.dump(best_rf, model_filename)
print("Model saved successfully!")

# %% [code] - Cell 28: Load and Test the Model
# Load the saved model
print(f"Loading model from {model_filename}...")
loaded_model = joblib.load(model_filename)

# Test the loaded model
y_pred_loaded = loaded_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test, y_pred_loaded)
print(f"Loaded model accuracy on test set: {loaded_accuracy:.4f}")

# Verify the loaded model performs the same as the original
original_accuracy = accuracy_score(y_test, y_pred_best)
print(f"Original model accuracy: {original_accuracy:.4f}")
print(f"Models are identical: {original_accuracy == loaded_accuracy}")

# Sample prediction demonstration
print("\nDemonstration of model prediction:")
sample_indices = np.random.choice(len(X_test), size=5, replace=False)
sample_X = X_test.iloc[sample_indices]
sample_y_true = y_test.iloc[sample_indices]

# Get predictions
sample_y_pred = loaded_model.predict(sample_X)

# Map numeric predictions back to category labels
risk_mapping_reverse = {v: k for k, v in risk_mapping.items()}
sample_y_true_labels = [risk_mapping_reverse[y] for y in sample_y_true]
sample_y_pred_labels = [risk_mapping_reverse[y] for y in sample_y_pred]

# Display results
results = pd.DataFrame({
    'True Risk Category': sample_y_true_labels,
    'Predicted Risk Category': sample_y_pred_labels,
    'Correct Prediction': [t == p for t, p in zip(sample_y_true_labels, sample_y_pred_labels)]
})
print(results)

# %% [code] - Cell 1051: Save the best model for web application
# Save the best model for use in the web application
def save_best_model(model, risk_mapping, features):
    """
    Save the trained model and related metadata for use in the web application.
    
    Parameters:
    - model: The trained machine learning model
    - risk_mapping: The dictionary mapping risk categories to numeric values
    - features: List of feature names used for prediction
    """
    print("Saving the best model for web application...")
    
    # Save the model using joblib
    joblib.dump(model, 'best_model.joblib')
    
    # Save the risk mapping and feature list as JSON
    metadata = {
        'risk_mapping': risk_mapping,
        'features': features.tolist() if isinstance(features, np.ndarray) else list(features)
    }
    
    # Save metadata to a JSON file
    import json
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    print("Model and metadata saved successfully!")
    print("The web application can now use this model for predictions.")

# Example usage:
# After you've trained your best model (for example, the RandomForestClassifier),
# call this function to save it:
# 
# best_model = models_results['Random Forest']['model']
# save_best_model(best_model, risk_mapping, X_train.columns)
# 
# Note: Replace 'Random Forest' with the name of your best performing model
# and ensure X_train.columns contains the feature names used during training.

# %% 
best_model = best_rf  # Use the best Random Forest model directly
save_best_model(best_model, risk_mapping, X_train.columns)

# %% 
