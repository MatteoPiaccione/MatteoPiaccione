# Baisc Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import time
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, auc, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

############### Data Manipulation and Data Cleaning ###############

# Load df
df = pd.read_csv('data/hospital_readmissions.csv')

df.info()
df.isna().sum()
df.duplicated().sum()

# Check for cat variables and and special characters
for column, values in df.iteritems():
    unique_values = values.sort_values().unique()
    print(f"Unique values in column '{column}': {unique_values}")

num = df.select_dtypes(exclude=['object'])

# Plot data distribution
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes[-1, -1].remove()
sns.set_style('dark')
sns.set_palette('Blues_r')

for ax, col in zip(axes.flatten(), num.columns):
    sns.kdeplot(num[col], ax=ax, fill=True)
    ax.set_title(col, fontsize=15)

plt.subplots_adjust(hspace=0.3, wspace=0.3)

plt.show()

# Basic statistic
df.describe()

# Plot Basic statistic and outliers
print('Figure 13. Subplot distribution and outliers.')
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes[-1, -1].remove()
sns.set_palette('bright')

# Iterate over the axes and the columns to fill the subplots with boxplots
for ax, col in zip(axes.flatten(), num.columns):
    ax.boxplot(num[col], medianprops={'color': 'mediumseagreen'})
    ax.set_title(col, fontsize=15)
    ax.set_ylabel('Value count')
    ax.set_xticks([])

plt.subplots_adjust(hspace=0.2, wspace=0.2)

plt.show()

# Inspecting outliers
Q1 = num.quantile(0.25)
Q3 = num.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = num[(num < lower_bound) | (num > upper_bound)]

print(
    f"The total number of outliers is: {outliers.count().sum()}\n\nNumber of outliers in each columns:\n{outliers.count()}")

# Feature correlation
df['readmitted'] = df.readmitted.map({'yes': 1, 'no': 0})
corr = df.corr()

# Plot correlation
fig = plt.figure(figsize=(9, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap='Blues', mask=mask)

plt.show()

# Ranking table diagnosis by age group
age_group = df.groupby(['age', 'diag_1']).size().reset_index(name='counts')

pivot_table = pd.pivot_table(age_group,
                             index='diag_1',
                             columns='age',
                             values='counts').drop('Missing', axis=0).rank(ascending=False, axis=0)


def color_rank_one(val):
    '''This function applies the orange background color to the first rank'''
    if val == 1:
        return 'background-color: mediumturquoise'
    else:
        return ''


pivot_table.style.format('{:,.0f}').background_gradient(cmap='Blues_r', axis=0).applymap(color_rank_one)

# Plot diagnosis by age group
fig, ax = plt.subplots(figsize=(10, 10))
sns.set_style('white')

diagnosis = df[df['diag_1'] != 'Missing']
unique_diags = diagnosis.diag_1.unique()

blues = sns.color_palette('Blues', n_colors=len(unique_diags))
custom_palette = ["mediumturquoise" if diag == "Circulatory"
                  else blues[len(unique_diags) - 1 - i]
                  for i, diag in enumerate(unique_diags)]

sns.countplot(y='age',
              hue='diag_1',
              order=diagnosis.age.sort_values().unique(),
              palette=custom_palette,
              data=diagnosis
              )

for bar in ax.patches:
    width = bar.get_width()
    x = width
    y = bar.get_y() + bar.get_height() / 2
    label = f"{width:.0f}"
    ax.annotate(label, (x, y),
                ha='left', va='center',
                xytext=(3, 0), fontsize=10,
                textcoords='offset points'
                )

sns.despine(bottom=True)
plt.ylabel('Age Group', fontsize=12)
plt.xlabel('')
plt.xticks([])
plt.title('Count primary diagnosis by age group')

plt.show()

# Maps the values in the 'readmitted' column
df['readmitted'] = df.readmitted.map({1: 'yes', 0: 'no'})

# Filtering diag_1 equal diabetes
df_diag_1 = df[df['diag_1'] == 'Diabetes']

# Table readmission rate by diag_1 equal diabetes
table_1 = (df_diag_1['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission_rate')
           .rename(columns={'index': 'readmitted'})
           )

table_1

# Filtering diag_2 equal diabetes
df_diag_2 = df[df['diag_2'] == 'Diabetes']

# Table readmission rate by diag_2 equal diabetes
table_2 = (df_diag_2['readmitted']
           .value_counts(normalize=True)
           .mul(100).round()
           .reset_index(name='readmission_rate')
           .rename(columns={'index': 'readmitted'})
           )

table_2

# Filtering diag_1 equal diabetes
df_diag_3 = df[df['diag_3'] == 'Diabetes']

# Table readmission rate by diag_3 equal diabetes
table_3 = (df_diag_3['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission_rate')
           .rename(columns={'index': 'readmitted'})
           )

table_3

# Filtering df different by 'Diabetes'
other_diag = df[df.apply(lambda x: 'Diabetes' not in x.values, axis=1)]

# Df other diagnosis
other_diag = (other_diag[['readmitted', 'diag_1', 'diag_2', 'diag_3']]
              .melt(id_vars='readmitted', var_name='diag', value_name='diseases')
              .drop('diag', axis=1)
              )

# Table readmission rate for each of the other diagnoses.
other_readmission = (other_diag.
                     groupby('diseases')
                     .value_counts(normalize=True)
                     .mul(100)
                     .round()
                     .reset_index(name='readmission_rate')
                     )

other_readmission = other_readmission[other_readmission['diseases'] != 'Missing']

# Table readmission by other diagnosis
toal_other_readmission = (other_readmission
                          .groupby('readmitted')['readmission_rate']
                          .mean()
                          .reset_index(name='readmission_rate')
                          )

toal_other_readmission

# Concat  table_1, table_2, table_3
diabetes_diag = pd.concat([table_1, table_2, table_3])

# Table diabetes_diag
diabetes_diag = pd.DataFrame(diabetes_diag
                             .groupby('readmitted')['readmission_rate']
                             .mean()
                             .reset_index()
                             )

# Table diabetes_other_diag
diabetes_other_diag = pd.concat([diabetes_diag, toal_other_readmission])

# Filtering fro readmitted = yes
diabetes_other_diag = diabetes_other_diag[diabetes_other_diag['readmitted'] == 'yes']

# Create a new column
diabetes_other_diag['diagnosis'] = ['Diabetes', 'Other']

# Sorting columns
diabetes_other_diag = diabetes_other_diag.reindex(columns=['diagnosis', 'readmitted', 'readmission_rate'])

diabetes_other_diag

# Plot Readmission Rate by Diabete and Other Diagnosises
sns.set_style('whitegrid')
sns.set_palette('Paired')
fig, ax = plt.subplots(figsize=(8, 6))

sns.barplot(x='diagnosis', y='readmission_rate', data=diabetes_other_diag)

for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(f"{height:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 1), textcoords='offset points',
                ha='center', va='bottom', fontsize=12)

sns.despine(left=True)
plt.title('Readmission Rate by Diabete and Other Diagnosises\nfor Patients Readmitted', fontsize=15)
plt.xlabel('Diagnosises', fontsize=12)
plt.ylabel('Readmission Rate', fontsize=12)
plt.show()

############### Chi-squared test ###############
# Set the p-value threshold to 0.05
alpha = 0.05

# Create a new column 'has_diabetes' indicating whether the primary diagnosis is diabetes
df['has_diabetes'] = df.diag_1.str.contains('Diabetes')

# Perform a chi-squared independence test between 'has_diabetes' and 'readmitted' and obtain the p-value
expected, observed, stats = pg.chi2_independence(data=df, y='has_diabetes', x='readmitted', correction=False)

df.drop('has_diabetes', axis=1, inplace=True)

# Print results
print(
    '    ---------------------------------------    Results Chi-squared test    ---------------------------------------\n')
print(f'The p_value is: {stats.pval.max()} \nIs p_value less than alpha?: {stats.pval.max() < alpha}')

##Other table
# table Readmission vs glucose test
table_4 = (df
           .groupby('glucose_test')['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission_rate')
           )

table_4

# Table readmission vs AC1 test
table_5 = (df
           .groupby('A1Ctest')['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission_rate')
           )

table_5

# Table Readmission vs diabetes medication
table_6 = (df
           .groupby('diabetes_med')['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission_rate')
           )

table_6

# Table readmission vs change medication
table_7 = (df
           .groupby('change')['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission_rate')
           )

table_7

# Table readmission vs time in hospital
table_8 = (df
           .groupby('time_in_hospital')['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission_rate')
           )

table_8

# Plot Readmission Rates by Each Diabetes Diagnosis
sns.set_style('whitegrid')
sns.set_palette('Paired')

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 4))

for i, table in enumerate([table_1, table_2, table_3]):
    ax = axes[i]
    sns.barplot(x='readmitted', y='readmission_rate', data=table.sort_values('readmitted'), ax=ax)
    ax.set_title(f'diag_{i + 1}')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Annotate bar plot with percentage values
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(f"{height:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12)

fig.suptitle("Readmission Rates by Each Diabetes Diagnosis", fontsize=15)
fig.text(0.5, -0.02, 'Readmission', ha='center', fontsize=13)
fig.text(0, 0.5, 'Percentages', va='center', fontsize=13, rotation=90)
plt.tight_layout()
sns.despine(left=True)

plt.show()

# Subplots Other Tables
sns.set_style('whitegrid')
fig, axes = plt.subplots(5, 2, sharey='row', figsize=(20, 25))
axes[-1, -1].remove()

# Plot Readmisson vs Glucose Test
ax = sns.barplot(x='glucose_test',
                 y='readmission_rate',
                 hue='readmitted',
                 order=['high', 'normal', 'no'],
                 hue_order=['no', 'yes'],
                 data=table_4,
                 ax=axes[0, 0]
                 )

for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.0f}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(0, 6), fontsize=12,
                textcoords='offset points')

ax.legend(title='Readmission', loc='upper right', bbox_to_anchor=(1.16, 1))
ax.set_xlabel('Glucose Test', fontsize=13)
ax.set_ylabel('Readmission Rates', fontsize=13)
ax.set_title('Readmission Rate by Patients with Glucose Test', fontsize=15)
sns.despine(left=True)

# Plot Readmisson vs AC1 Test
ax = sns.barplot(x='A1Ctest',
                 y='readmission_rate',
                 hue='readmitted',
                 order=['high', 'normal', 'no'],
                 data=table_5,
                 ax=axes[0, 1])

for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.0f}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(0, 6), fontsize=12,
                textcoords='offset points')

ax.legend().set_visible(False)
ax.set_xlabel('A1C Test', fontsize=13)
ax.set_ylabel('')
ax.set_title('Readmission Rate by Patients with A1C Test', fontsize=15)
sns.despine(left=True)

# Plot Radmission vs Medication
data_yes = table_6[table_6['diabetes_med'] == 'yes']
ax = sns.barplot(x='readmitted', y='readmission_rate', data=data_yes, ax=axes[1, 0])
ax.set_title('Readmission Rate by Patients \n With Diabetes Medication', fontsize=15)
ax.set_xlabel('Readmission', fontsize=13)
ax.set_ylabel('Readmission Rates', fontsize=13)

for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.0f}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(0, 6), fontsize=12,
                textcoords='offset points')

sns.despine(left=True)

# Plot Readmission vs No Medication
data_no = table_6[table_6['diabetes_med'] == 'no']
ax = sns.barplot(x='readmitted', y='readmission_rate', data=data_no, ax=axes[1, 1])
ax.set_title('Readmission Rate by Patients \n Without Diabetes Medication', fontsize=15)
ax.set_xlabel('Readmission', fontsize=13)
ax.set_ylabel('')

for bar in ax.patches:
    ax.annotate(f'{bar.get_height()}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(0, 6), fontsize=12,
                textcoords='offset points')

# Plot Readmission vs Change Medication
data_yes = table_7[table_7['change'] == 'yes']
ax = sns.barplot(x='readmitted', y='readmission_rate', data=data_yes, ax=axes[2, 0])
ax.set_title('Readmission Rate by Patients \n With Change Diabetes Medication', fontsize=15)
ax.set_xlabel('Readmission', fontsize=13)
ax.set_ylabel('Readmission Rates', fontsize=13)

for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.0f}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(0, 6), fontsize=12,
                textcoords='offset points')

sns.despine(left=True)

# Plot Readmission vs No Change Medication
data_no = table_7[table_7['change'] == 'no']
ax = sns.barplot(x='readmitted', y='readmission_rate', data=data_no, ax=axes[2, 1])
ax.set_title('Readmission Rate by Patients \n Without Change Diabetes Medication', fontsize=15)
ax.set_xlabel('Readmission', fontsize=13)
ax.set_ylabel('')

for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.0f}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(0, 6), fontsize=12,
                textcoords='offset points')

sns.despine(left=True)

# Plot readmission rate by other diagnosis
ax = sns.barplot(x='diseases',
                 y='readmission_rate',
                 hue='readmitted',
                 data=other_readmission,
                 ax=axes[3, 0])

for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.0f}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(2, 6), fontsize=12,
                textcoords='offset points')

ax.set_title('Readmission rate by other diagnosis', fontsize=15)
ax.set_ylabel('Readmission Rate', fontsize=13)
ax.set_xlabel('')
ax.legend(title='Readmitted', loc='upper right', bbox_to_anchor=(1.15, 1))
sns.despine(left=True)

# Polt Readmission vs Time in hospital
ax = sns.barplot(x='time_in_hospital',
                 y='readmission_rate',
                 hue='readmitted',
                 data=table_8,
                 ax=axes[3, 1]
                 )
ax.axvspan(xmin=5.53,
           xmax=9.53,
           ymax=0.85,
           facecolor='mediumaquamarine',
           edgecolor='mediumaquamarine',
           alpha=0.4,
           lw=1.5,
           zorder=-1
           )
ax.axvspan(xmin=10.53,
           xmax=11.50,
           ymax=0.85,
           facecolor='mediumaquamarine',
           edgecolor='mediumaquamarine',
           alpha=0.4,
           lw=1.5,
           zorder=-1
           )

ax.set_ylabel('')
ax.set_xlabel('Days in Hospital', fontsize=13)
ax.set_title('Readmission Rates by Days in Hospital', fontsize=15)
ax.legend().set_visible(False)
sns.despine(left=True)

# correlation heatmap
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap='Blues', mask=mask, ax=axes[4, 0])

plt.subplots_adjust(hspace=0.4, wspace=0.18)
plt.title('Correlation heatmap', fontsize=15)

plt.show()

# Maps the values in the 'readmitted' column
df['readmitted'] = df['readmitted'].map({'yes': 1, 'no': 0})

# Find the readmission_rate mean
tot_mean = round((df.readmitted.mean() * 100))

# Table readmission rate by age group
hp_by_age = (df.groupby('age')['readmitted']
             .mean()
             .mul(100)
             .round()
             .reset_index(name='readmission_rate')
             )

hp_by_age

# Plot Mean readmission rate by age group
fig = plt.figure(figsize=(8, 6))
sns.set_style('white')
sns.set_palette('Paired')

sns.lineplot(x='age', y='readmission_rate', lw=2, data=hp_by_age)

plt.axhline(y=tot_mean, color='mediumaquamarine', linestyle='--', )

plt.text(6.5, tot_mean,
         f'Average Readmission Rate: {tot_mean}%',
         ha='center', va='center',
         color='mediumaquamarine', fontsize=13
         )

plt.title('Readmission Rate by Age Group', fontsize=15)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Mean Readmission Rate', fontsize=12)
sns.despine()

plt.show()

# Table readmission rate by diagnosis
hp_by_diag = (df[df.diag_1 != "Missing"]
              .groupby('diag_1')['readmitted']
              .mean()
              .mul(100)
              .round()
              .reset_index(name='readmission_rate')
              )

hp_by_diag

# Plot readmission rate by diagnosis
fig = plt.figure(figsize=(7, 6))
sns.set_style('white')

sns.barplot(
    y='readmission_rate',
    x='diag_1',
    palette='Blues_r',
    data=hp_by_diag.sort_values('readmission_rate', ascending=False)
)

plt.axhline(y=tot_mean, color='mediumaquamarine', linestyle='--')

plt.text(
    8, tot_mean,
    f'Average Readmission Rate: {tot_mean}%',
    ha='center', va='center', color='mediumaquamarine'
)

plt.title('Readmission Rate by Diagnosis')
plt.ylabel('Mean Readmission Rate')
plt.xlabel('')
plt.xticks(rotation=90)
sns.despine(left=True)

plt.show()

############### MACHINE LEARNING ###############
###Feature engenieering and preprocessing###
seed = 217

# Mapping target variable
df['readmitted'] = df.readmitted.replace({'yes': 1, 'no': 0})

# Create feature and target set
X = df.drop('readmitted', axis=1)
y = df.readmitted

# Splitting feature and target set into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

# Selecting categorical and numeric columns
num_selector = make_column_selector(dtype_exclude='object')
cat_selector = make_column_selector(dtype_include='object')
num_cols = num_selector(X)
cat_cols = cat_selector(X)

# Preprocessing categorical and numeric columns
num_preprocessor = StandardScaler()
cat_preprocessor = OneHotEncoder()

# Merge preprocessed categorical and numeric columns
preprocessor = ColumnTransformer([
    ('StandardScaler', num_preprocessor, num_cols),
    ('One Hot Encoder', cat_preprocessor, cat_cols)
])

# Make a pipelines dictionary
pipelines = {'K-Neighbors Classifier': make_pipeline(preprocessor, KNeighborsClassifier(n_neighbors=13)),
             'Logistic Regression': make_pipeline(preprocessor, LogisticRegression(random_state=seed)),
             'Random Forest Classifier': make_pipeline(preprocessor, RandomForestClassifier(random_state=seed))}

# Defining a function to get a table of model metrics.
# Function to get models metrics
def metrics_from_pipes(pipes_dict):
    '''
    This function takes as input a dictionary of ML pipilines  and
    returns a table all the train and test metrics for
    each model in the dictionary
    '''
    train_accs = []
    train_f1s = []
    train_roc_aucs = []
    train_pr_aucs = []
    train_precs = []
    train_recs = []
    train_specs = []
    train_fprs_list = []
    train_fnrs_list = []

    test_accs = []
    test_f1s = []
    test_roc_aucs = []
    test_pr_aucs = []
    test_precs = []
    test_recs = []
    test_specs = []
    test_fprs_list = []
    test_fnrs_list = []

    for name, pipeline in pipes_dict.items():
        pipeline.fit(X_train, y_train)
        y_pred_test = pipeline.predict(X_test)
        y_pred_train = pipeline.predict(X_train)

        y_probs_test = pipeline.predict_proba(X_test)[:, 1]
        y_probs_train = pipeline.predict_proba(X_train)[:, 1]

        train_precisions, train_recalls, threshold = precision_recall_curve(y_train, y_probs_train)
        test_precisions, test_recalls, threshold = precision_recall_curve(y_test, y_probs_test)

        tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_pred_train).ravel()
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()

        train_accs.append(accuracy_score(y_train, y_pred_train))
        train_f1s.append(f1_score(y_train, y_pred_train))
        train_roc_aucs.append(roc_auc_score(y_train, y_probs_train))
        train_pr_aucs.append(auc(train_recalls, train_precisions))
        train_precs.append(precision_score(y_train, y_pred_train))
        train_recs.append(recall_score(y_train, y_pred_train))
        train_specs.append(tn_train / (tn_train + fp_train))
        train_fprs_list.append(fp_train / (fp_train + tn_train))
        train_fnrs_list.append(fn_train / (fn_train + tp_train))

        test_accs.append(accuracy_score(y_test, y_pred_test))
        test_f1s.append(f1_score(y_test, y_pred_test))
        test_roc_aucs.append(roc_auc_score(y_test, y_probs_test))
        test_pr_aucs.append(auc(test_recalls, test_precisions))
        test_precs.append(precision_score(y_test, y_pred_test))
        test_recs.append(recall_score(y_test, y_pred_test))
        test_specs.append(tn_test / (tn_test + fp_test))
        test_fprs_list.append(fp_test / (fp_test + tn_test))
        test_fnrs_list.append(fn_test / (fn_test + tp_test))

    # aggregate the performance metric lists into seperate dataframes
    train_metrics = pd.DataFrame(
        {'model': pipes_dict.keys(),
         'accuracy': train_accs,
         'f1_score': train_f1s,
         'roc_auc': train_roc_aucs,
         'pr_auc': train_pr_aucs,
         'precision': train_precs,
         'recall': train_recs,
         'specificity': train_specs,
         'false_positive_rate': train_fprs_list,
         'false_negative_rate': train_fnrs_list})

    test_metrics = pd.DataFrame(
        {'model': pipes_dict.keys(),
         'accuracy': test_accs,
         'f1_score': test_f1s,
         'roc_auc': test_roc_aucs,
         'pr_auc': test_pr_aucs,
         'precision': test_precs,
         'recall': test_recs,
         'specificity': test_specs,
         'false_positive_rate': test_fprs_list,
         'false_negative_rate': test_fnrs_list})

    # Merging metrics from train and test set
    train_test_metrics = train_metrics.merge(test_metrics,
                                             on='model',
                                             how='left',
                                             suffixes=('_train', '_test'))

    # Sorting coulmns
    train_test_metrics = train_test_metrics.reindex(columns=['model',
                                                             'accuracy_train',
                                                             'accuracy_test',
                                                             'f1_score_train',
                                                             'f1_score_test',
                                                             'roc_auc_train',
                                                             'roc_auc_test',
                                                             'pr_auc_train',
                                                             'pr_auc_test',
                                                             'precision_train',
                                                             'precision_test',
                                                             'recall_train',
                                                             'recall_test',
                                                             'specificity_train',
                                                             'specificity_test',
                                                             'false_positive_rate_train',
                                                             'false_positive_rate_test',
                                                             'false_negative_rate_train',
                                                             'false_negative_rate_test'])

    return train_test_metrics.set_index('model').transpose()

# Getting metrics_table
metrics_table = metrics_from_pipes(pipelines)
print('Table 2: Base models metrics table.')
metrics_table.style.background_gradient(cmap='Blues')

# Evaluating multiple models
def boxplot_cv_perfomances_from_pipes(pipelines_dict):
    results = []

    for pipeline in pipelines_dict.values():
        kf = KFold(n_splits=5)
        cv_results = cross_val_score(pipeline, X_train, y_train, cv=kf)
        results.append(cv_results)

    #Plot Cross-Validation Performance
    sns.set_style('dark')
    fig = plt.figure(figsize=(10,8))
    plt.boxplot(results, labels=pipelines.keys(), medianprops={'color': 'mediumseagreen'})
    plt.ylabel('Accurancy score', fontsize=12)
    #plt.title('Cross-Validation Performance for Three Models:\n KNN, Logistic Regression, and Random Forest Classifier')

    plt.show()

    print('\n------------------------------------------------------------------')

    #test set performance
    for name, pipeline in pipelines_dict.items():
        pipeline.fit(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        print(f"{name} Test Set Accuracy: {test_score}")

print('Figure 5: Cross-Validation performance for K-Neighbors Classifier, Logistic Regression, Random Forest Classifier.')
boxplot_cv_perfomances_from_pipes(pipelines)

# Defining a function to plot roc_pr_auc_curve.
def roc_pr_auc_curves_from_pipes(pipes_dict):
    fprss = []
    tprss = []
    precs = []
    recs = []
    roc_aucs = []
    pr_aucs = []

    for name, pipeline in pipes_dict.items():
        y_probs = pipeline.predict_proba(X_test)[:, 1]

        fprs, tprs, _ = roc_curve(y_test, y_probs)
        precisions, recalls, _ = precision_recall_curve(y_test, y_probs)

        fprss.append(fprs)
        tprss.append(tprs)
        precs.append(precisions)
        recs.append(recalls)
        roc_aucs.append(roc_auc_score(y_test, y_probs))
        pr_aucs.append(auc(recalls, precisions))

    sns.set_style('white')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for i in range(len(fprss)):
        model_name = list(pipes_dict.keys())[i]
        label_model = f"{model_name}\nAUC = {roc_aucs[i]:.3f}"
        ax1.plot(fprss[i], tprss[i], label=label_model)
    ax1.plot([0, 1], [0, 1], linestyle='--', label='Base rate\nAUC = 0.5', color='black')
    ax1.set_xlabel('False positive rate', fontsize=12)
    ax1.set_ylabel('True positive rate', fontsize=12)
    ax1.set_title('Test set area under the ROC curve', fontsize=15)
    ax1.legend(loc="lower right", bbox_to_anchor=(1, 0), ncol=2, frameon=True)

    for i in range(len(precs)):
        model_name = list(pipes_dict.keys())[i]
        label_model = f"{model_name}\nAUC = {pr_aucs[i]:.3f}"
        ax2.plot(recs[i], precs[i], label=label_model)
    ax2.plot([0, 1], [0, 0], linestyle='--', label='Base rate', color='black')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Test set area under the PR curve', fontsize=15)
    ax2.legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=2, frameon=True)
    plt.subplots_adjust(hspace=0.3)
    sns.despine()

print('Figure 6: Test set area under the ROC curve and PR curve')
roc_pr_auc_curves_from_pipes(pipelines)

# Setting start time to evaluete timing perfomance
start_time = time.time()

#Create a dictionary if hyperparameters
param_grid = {'logisticregression__solver': ['libliear', 'sag', 'saga'],
              'logisticregression__penalty' : ['l1', 'l2'],
              'logisticregression__C' : [0.1, 1.0, 10],
              'logisticregression__class_weight' : [None, 'balanced'],
              'logisticregression__max_iter' : [100, 400, 800]}

#Define a cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

#Instantiate a GridSearchCV object
grid_lr= GridSearchCV(estimator=pipelines['Logistic Regression'],
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=cv,
                            n_jobs=-1)

#Fit the GridSearchCV object
grid_lr.fit(X_train, y_train)

#Measure the execution time
end_time = time.time()
execution_time = (end_time - start_time)/60

print(f'The best estimator is: {grid_lr.best_estimator_}\n'
      f'The best params are: {grid_lr.best_params_}\n'
      f'The best score is: {grid_lr.best_score_}\n'
      f'Execution time: {execution_time:.3f} minutes')

# Plot roc_auc_curve fro best estimetor and from predictions
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_style('white')


# Function to get metrics from tuned model
def metrics_test_from_tuned(model):
    ''' This function take as input a hyperparameter model
    (i.e GridSerachCV, RandomizedSearchCV) and
    returns a table all the test metrics
    '''
    test_accs = []
    test_f1s = []
    test_roc_aucs = []
    test_pr_aucs = []
    test_precs = []
    test_recs = []
    test_specs = []
    test_fprs_list = []
    test_fnrs_list = []

    mod = model.best_estimator_
    y_pred_test = mod.predict(X_test)

    y_probs_test = mod.predict_proba(X_test)[:, 1]

    test_precisions, test_recalls, threshold = precision_recall_curve(y_test, y_probs_test)

    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()

    test_accs.append(accuracy_score(y_test, y_pred_test))
    test_f1s.append(f1_score(y_test, y_pred_test))
    test_roc_aucs.append(roc_auc_score(y_test, y_probs_test))
    test_pr_aucs.append(auc(test_recalls, test_precisions))
    test_precs.append(precision_score(y_test, y_pred_test))
    test_recs.append(recall_score(y_test, y_pred_test))
    test_specs.append(tn_test / (tn_test + fp_test))
    test_fprs_list.append(fp_test / (fp_test + tn_test))
    test_fnrs_list.append(fn_test / (fn_test + tp_test))

    test_metrics = pd.DataFrame(
        {'model': [model.best_estimator_.steps[-1][1].__class__.__name__],
         'accuracy': test_accs,
         'f1_score': test_f1s,
         'roc_auc': test_roc_aucs,
         'pr_auc': test_pr_aucs,
         'precision': test_precs,
         'recall': test_recs,
         'specificity': test_specs,
         'false_positive_rate': test_fprs_list,
         'false_negative_rate': test_fnrs_list})

    return test_metrics.transpose().reset_index().rename(columns={'index': 'metrics', 0: 'values'})

# Getting metrics_table
metrics_table = metrics_test_from_tuned(grid_lr)
print('Table 3: Best estimator metrics table.')
metrics_table

# Plot roc_pr_auc curves from best estimator
def roc_pr_auc_curves_from_best_estimator(model):
    '''This function takes as input an hyperparameter model and
    returns best estimator ROC_PR AUC curves'''
    precs = []
    recs = []
    pr_aucs = []

    mod = model.best_estimator_
    y_pred = model.predict(X_test)
    y_probs = mod.predict_proba(X_test)[:, 1]

    precisions, recalls, _ = precision_recall_curve(y_test, y_probs)
    precs.append(precisions)
    recs.append(recalls)
    pr_aucs.append(auc(recalls, precisions))

    sns.set_style('white')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    roc_best_estimator = RocCurveDisplay.from_estimator(mod, X_test, y_test, name='Best estimator\n', ax=ax1)
    roc_from_predictions = RocCurveDisplay.from_predictions(y_test, y_pred, name='From predictions\n', ax=ax1)
    ax1.plot([0, 1], ls="--", color='k', label='Base rate\n(AUC = 0.5)')
    ax1.set_title('Test set area under the ROC curve', fontsize=15)
    ax1.legend(loc='lower right', bbox_to_anchor=(1, 0))
    sns.despine()

    for i in range(len(precs)):
        model_name = model.best_estimator_.steps[-1][1].__class__.__name__
        label_model = f"{model_name}\nAUC = {pr_aucs[i]:.3f}"
        ax2.plot(recs[i], precs[i], label=label_model)
    ax2.plot([0, 1], [0, 0], linestyle='--', label='Base rate', color='black')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Test set area under the PR curve', fontsize=15)
    ax2.legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=2, frameon=True)
    plt.subplots_adjust(hspace=0.3)
    sns.despine()

print('Figure 7: Best model ROC and PR curves')
roc_pr_auc_curves_from_best_estimator(grid_lr)

#Predict target variable using the best estimator
y_pred_grid_lr = grid_lr.predict(X_test)

#Create a Confusion Matrix Display object
cm =ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_grid_lr),
                          display_labels=['Not readmitted', 'Readmitted'])

print('Figure 8: GridSearchCV best estimator (Logistic Regression) confusion matrix.')
cm.plot(cmap='Blues')
plt.show()

# Get coefficients of the best Logistic Regression estimator
print('Figure 9: GridSearchCV best estimator (Logistic Regression) coefficient importances.')
coeffs = grid_lr.best_estimator_.named_steps.logisticregression.coef_[0]

#Sort coefficents
importances_model = pd.Series(coeffs[:len(X.columns)],
                              index = X
                              .columns[:len(coeffs[:len(X.columns)])]).sort_values()

#Plot  LogisticRegression coefficients
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(10,7))
palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)

importances_model.plot(kind='barh', color=palette(importances_model/float(importances_model.max())))
plt.xlabel('Features', fontsize=12)
plt.ylabel('Features importances', fontsize=12)

plt.show()