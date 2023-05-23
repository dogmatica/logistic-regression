#!/usr/bin/env python
# coding: utf-8

# ---
# 
# # **Part I: Research Question**
# 
# ## Research Question
# 
# My dataset for this predictive modeling exercise includes data on an internet service provider’s current and former subscribers, with an emphasis on customer churn (whether customers are maintaining or discontinuing their subscription to the ISP’s service).  Data analysis performed on the dataset will be aimed with this research question in mind: is there a relationship between customer lifestyle, or “social” factors, and customer churn?  Lifestyle and social factors might include variables such as age, income, and marital status, among others.

# ---
# 
# ## Objectives and Goals
# 
# Conclusions gleaned from the analysis of this data can benefit stakeholders by revealing information on which customer populations may be more likely to “churn”, or terminate their service contract with the ISP.  Such information may be used to fuel targeted advertising campaigns, special promotional offers, and other strategies related to customer retention.

# ---
# 
# # **Part II: Method Justification**
# 
# ## Assumptions of a logistic regression model
# 
# The assumptions of a logistic regression model are as follows:
# 
# - The Response Variable is Binary
# - The Observations are Independent
# - There is No Multicollinearity Among Explanatory Variables
# - There are No Extreme Outliers
# - There is a Linear Relationship Between Explanatory Variables and the Logit of the Response Variable
# - The Sample Size is Sufficiently Large
# 
# For each of these assumptions that are violated, the potential reliability of the logistic regression model decreases. Adherence to these assumptions can be measured via tests such as Box-Tidwell, checking for extreme outliers, and VIF (Zach, 2021).

# ---
# 
# ## Tool Selection
# 
# All code execution was carried out via Jupyter Lab, using Python 3.  I used Python as my selected programming language due to prior familiarity and broader applications when considering programming in general.  R is a very strong and robust language tool for data analysis and statistics but finds itself somewhat limited to that niche role (Insights for Professionals, 2019).  I utilized the NumPy, Pandas, and Matplotlib libraries to perform many of my data analysis tasks, as they are among the most popular Python libraries employed for this purpose and see widespread use.  Seaborn is included primarily for its better-looking boxplots, seen later in this document (Parra, 2021).  
# 
# Beyond these libraries, I relied upon the Statsmodels library.  Statsmodels is one of several Python libraries that support linear and logistic regression.  I am most familiar with it due to the course material's heavy reliance upon it.  I also used the confusion_matrix and accuracy_score functions from scikit-learn's metrics module.

# In[ ]:


# Imports and housekeeping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from sklearn.metrics import (confusion_matrix, accuracy_score)
sns.set_theme(style="darkgrid")


# ---
# 
# ## Why Logistic Regression?
# 
# Like linear regression, logistic regression is used to understand the relationship between one or more independent variables and a single dependent variable.  Where logistic regression differs is in the type of prediction being made; where linear regression better helps us predict measurements, logistic regression helps predict whether or not an event will occur or a particular choice will be made.  It works best when used with a dependent variable that has an “either/or” or “yes/no” response.  Utilizing multiple independent variables in a predictive model can make our predictions stronger and allows higher conviction in the reliance on those models for decision making.
# 

# ---
# 
# # **Part III: Data Preparation**
# 
# ## Data Preparation Goals and Data Manipulations
# 
# I would like my data to include only variables relevant to my research question, and to be clean and free of missing values and duplicate rows.  It will also be important to re-express any categorical variable types with numeric values.  My first steps will be to import the complete data set and execute functions that will give me information on its size, the data types of its variables, and a peek at the data in table form.  I will then narrow the data set to a new dataframe containing only the variables I am concerned with, and then utilize functions to determine if any null values or duplicate rows exist.

# In[ ]:


# Import the main dataset
df = pd.read_csv('churn_clean.csv',dtype={'locationid':np.int64})


# In[ ]:


# Display dataset info
df.info()


# In[ ]:


# Display dataset top 5 rows
df.head()


# In[ ]:


# Trim dataset to variables relevant to research question
columns = ['Area', 'Children', 'Age', 'Income', 'Marital', 'Gender', 'Churn', 'Outage_sec_perweek', 
           'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year']
df_data = pd.DataFrame(df[columns])


# In[ ]:


# Check data for null or missing values
df_data.isna().any()


# In[ ]:


# Check data for duplicated rows
df_data.duplicated().sum()


# ---
# 
# ## Summary Statistics
# 
# I can use the describe() function to display the summary statistics for the entire dataframe, as well as each variable I'll be evaluating for inclusion in the model.  I have selected the Churn variable as my dependent variable.
# 
# I will also utilize histogram plots to illustrate the distribution of each numeric variable in the dataframe, and countplots for the categorical variables.

# In[ ]:


# Display summary statistics for entire dataset - continuous variables
df_data.describe()


# In[ ]:


# Display summary statistics for entire dataset - categorical variables
df_data.describe(include = object)


# In[ ]:


# Initialize figure size settings
plt.rcParams['figure.figsize'] = [10, 10]


# In[ ]:


# Display histogram plots for distribution of continuous variables
df_data.hist()


# In[ ]:


# Display histogram plot and summary statistics for Bandwidth_GB_Year
df_data['Bandwidth_GB_Year'].hist(legend = True)
plt.show()
df_data['Bandwidth_GB_Year'].describe()


# In[ ]:


# Display histogram plot and summary statistics for Children
df_data['Children'].hist(legend = True)
plt.show()
df_data['Children'].describe()


# In[ ]:


# Display histogram plot and summary statistics for Age
df_data['Age'].hist(legend = True)
plt.show()
df_data['Age'].describe()


# In[ ]:


# Display histogram plot and summary statistics for Income
df_data['Income'].hist(legend = True)
plt.show()
df_data['Income'].describe()


# In[ ]:


# Display histogram plot and summary statistics for Outage_sec_perweek
df_data['Outage_sec_perweek'].hist(legend = True)
plt.show()
df_data['Outage_sec_perweek'].describe()


# In[ ]:


# Display histogram plot and summary statistics for Yearly_equip_failure
df_data['Yearly_equip_failure'].hist(legend = True)
plt.show()
df_data['Yearly_equip_failure'].describe()


# In[ ]:


# Display histogram plot and summary statistics for Tenure
df_data['Tenure'].hist(legend = True)
plt.show()
df_data['Tenure'].describe()


# In[ ]:


# Display histogram plot and summary statistics for MonthlyCharge
df_data['MonthlyCharge'].hist(legend = True)
plt.show()
df_data['MonthlyCharge'].describe()


# In[ ]:


# Display countplots for distribution of categorical variables
fig, ax = plt.subplots(figsize = (20,20), ncols = 2, nrows = 2)
sns.countplot(x='Area', data=df_data, ax = ax[0][0])
sns.countplot(x='Marital', data=df_data, ax = ax[0][1])
sns.countplot(x='Gender', data=df_data, ax = ax[1][0])
sns.countplot(x='Churn', data=df_data, ax = ax[1][1])


# In[ ]:


# Display countplot and summary statistics for Area
sns.countplot(x='Area', data=df_data)
plt.show()
df_data['Area'].describe()


# In[ ]:


# Display countplot and summary statistics for Marital
sns.countplot(x='Marital', data=df_data)
plt.show()
df_data['Marital'].describe()


# In[ ]:


# Display countplot and summary statistics for Gender
sns.countplot(x='Gender', data=df_data)
plt.show()
df_data['Gender'].describe()


# In[ ]:


# Display countplot and summary statistics for Churn
sns.countplot(x='Churn', data=df_data)
plt.show()
df_data['Churn'].describe()


# ---
# 
# ## Further Preparation Steps
# 
# I will make some adjustments to my data types to make my variables easier to work with.  Conversion of "object" types as "category" in particular will lend itself to a more efficient conversion of categorical variables to numeric.

# In[ ]:


# Reassign data types
for col in df_data:
    if df_data[col].dtypes == 'object':
        df_data[col] = df_data[col].astype('category')
    if df_data[col].dtypes == 'int64':
        df_data[col] = df_data[col].astype(int)
    if df_data[col].dtypes == 'float64':
        df_data[col] = df_data[col].astype(float)


# In[ ]:


# Display dataset info and observe data type changes
df_data.info()


# ---
# 
# Here I will use the cat.codes accessor to perform label encoding on my categorical variables.

# In[ ]:


# Use cat.codes for label encoding of 4 categorical variables
df_data['Area_cat'] = df_data['Area'].cat.codes
df_data['Marital_cat'] = df_data['Marital'].cat.codes
df_data['Gender_cat'] = df_data['Gender'].cat.codes
df_data['Churn_cat'] = df_data['Churn'].cat.codes


# In[ ]:


# Display dataset top 5 rows from label encoded variables
df_data[['Area', 'Marital', 'Gender', 'Churn', 'Area_cat', 'Marital_cat', 'Gender_cat', 'Churn_cat']].head()


# ---
# 
# ## Univariate and Bivariate Visualizations
# 
# Univariate analysis of each variable can be seen above in section 2 of part III, "Data Preparation".  I will make use of Seaborn's boxplot() function for bivariate analysis of all variables.  Each independent variable is paired against my dependent variable, "Churn".

# In[ ]:


# Display boxplots for bivariate analysis of variables - dependent variable = Churn
fig, ax = plt.subplots(figsize = (20, 20), ncols = 4, nrows = 3)
sns.boxplot(x = 'Churn', y = 'Children', data = df_data, ax = ax[0][0])
sns.boxplot(x = 'Churn', y = 'Age', data = df_data, ax = ax[0][1])
sns.boxplot(x = 'Churn', y = 'Income', data = df_data, ax = ax[0][2])
sns.boxplot(x = 'Churn', y = 'Outage_sec_perweek', data = df_data, ax = ax[0][3])
sns.boxplot(x = 'Churn', y = 'Yearly_equip_failure', data = df_data, ax = ax[1][0])
sns.boxplot(x = 'Churn', y = 'Tenure', data = df_data, ax = ax[1][1])
sns.boxplot(x = 'Churn', y = 'MonthlyCharge', data = df_data, ax = ax[1][2])
sns.boxplot(x = 'Churn', y = 'Bandwidth_GB_Year', data = df_data, ax = ax[1][3])
sns.boxplot(x = 'Churn', y = 'Area_cat', data = df_data, ax = ax[2][0])
sns.boxplot(x = 'Churn', y = 'Marital_cat', data = df_data, ax = ax[2][1])
sns.boxplot(x = 'Churn', y = 'Gender_cat', data = df_data, ax = ax[2][2])


# ---
# 
# ## Copy of Prepared Data Set
# 
# Below is the code used to export the prepared data set to CSV format.

# In[ ]:


# Export prepared dataframe to csv
df_data.to_csv(r'C:\Users\wstul\d208\churn_clean_perpared.csv')


# ---
# 
# # **Part IV: Model Comparison and Analysis**
# 
# ## Initial Logistic Regression Model
# 
# Below I will create an initial logistic regression model and display its summary info.

# In[ ]:


# Create initial model and display summary
mdl_churn_vs_all = logit("Churn_cat ~ Area_cat + Children + Age + Income + Marital_cat + Gender_cat + Bandwidth_GB_Year + \
                        Outage_sec_perweek + Yearly_equip_failure + MonthlyCharge + Tenure", data=df_data).fit()
print(mdl_churn_vs_all.summary())


# ---
# 
# ## Reducing the Initial Model
# 
# Starting from this initial model, I will aim to reduce the model by eliminating variables not suitable for this logistic regression, using statistical analysis in my selection process.
# 
# To begin I will look at some additional metrics for the current model.

# In[ ]:


# defining the dependent and independent variables
Xtest = df_data[['Area_cat', 'Children', 'Age', 'Income', 'Marital_cat', 'Gender_cat', 'Outage_sec_perweek', 
           'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year']]
ytest = df_data['Churn_cat']
# performing predictions on the test datdaset
yhat = mdl_churn_vs_all.predict(Xtest)
prediction = list(map(round, yhat))
# confusion matrix
conf_matrix = confusion_matrix(ytest, prediction)
print ("Confusion Matrix : \n", conf_matrix)
# accuracy score of the model
print('Test accuracy = ', accuracy_score(ytest, prediction))
# confusion matrix visualized
mosaic(conf_matrix)


# ---
# 
# As I proceed through my reduction process, I will aim to keep the test accuracy close to the initial model's performance while minimizing additional false positives and negatives.  Higher accuracy scores are considered better, with 1.000 being the maximum.
# 
# First I will generate a correlation table as a reference during the selection process, and perform a variance inflation factor analysis for all features currently in the model.

# In[ ]:


df_data.corr()


# In[ ]:


# Perform variance inflation factor analysis for initial feature set
X = df_data[['Area_cat', 'Children', 'Age', 'Income', 'Marital_cat', 'Gender_cat', 'Outage_sec_perweek', 
           'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year']]
vif_data = pd.DataFrame()
vif_data['IndVar'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)


# ---
# 
# Right away, I can see very high VIF scores for two variables, Tenure and Bandwidth_GB_Year.  High VIF values (usually greater than 5-10) indicate a high degree of multicollinearity with other variables in the model.  This reduces the model accuracy, so I will start by dropping one of these two variables from the set and repeat my VIF analysis.
# 
# As Tenure has a slightly better correlation with my dependent variable, Churn_cat, I will drop Bandwidth_GB_Year first.

# In[ ]:


# Drop 1 high VIF variable
X = X.drop('Bandwidth_GB_Year', axis = 1)


# In[ ]:


# Perform variance inflation factor analysis for trimmed feature set
vif_data = pd.DataFrame()
vif_data['IndVar'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)


# ---
# 
# The VIF scores look much better than they did, but there are still a few that are rather high.  Referring back to my correlation table, MonthlyCharge has a far greater correlation with my dependent variable than Outage_sec_perweek does, so I will drop Outage_sec_perweek and repeat the test.

# In[ ]:


# Drop 1 high VIF variable
X = X.drop('Outage_sec_perweek', axis = 1)


# In[ ]:


# Perform variance inflation factor analysis for trimmed feature set
vif_data = pd.DataFrame()
vif_data['IndVar'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)


# ---
# 
# I have only 2 variables remaining with VIF greater than 5.  Once again the correlation table recognizes MonthlyCharge as a better candidate for inclusion in the model, so Age will be dropped from the group of independent variables.

# In[ ]:


# Drop 1 high VIF variable
X = X.drop('Age', axis = 1)


# In[ ]:


# Perform variance inflation factor analysis for trimmed feature set
vif_data = pd.DataFrame()
vif_data['IndVar'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)


# ---
# 
# While MonthlyCharge still has a score higher than the other remaining variables, it is still less than 10.
# 
# I will create a reduced model based on my remaining variables to see how our statistics look.

# In[ ]:


# Create first reduced model and display summary
mdl_churn_vs_reduced = logit("Churn_cat ~ Area_cat + Children + Income + Marital_cat + Gender_cat + Yearly_equip_failure + MonthlyCharge + Tenure",
                           data=df_data).fit()
print(mdl_churn_vs_reduced.summary())


# ---
# 
# According to this summary, several variables exhibit high p-values, indicating no relationship between that variable and the dependent variable, Churn_cat.  A value greater than .05 is considered high.  I will remove these variables from the model and once again evaluate the resulting summary and statistics.

# In[ ]:


# Create first reduced model and display summary
mdl_churn_vs_features = logit("Churn_cat ~ MonthlyCharge + Tenure",
                           data=df_data).fit()
print(mdl_churn_vs_features.summary())


# In[ ]:


Xtest = df_data[['MonthlyCharge', 'Tenure']]
ytest = df_data['Churn_cat']
yhat = mdl_churn_vs_features.predict(Xtest)
prediction = list(map(round, yhat))
conf_matrix = confusion_matrix(ytest, prediction)
print ("Confusion Matrix : \n", conf_matrix)
print('Test accuracy = ', accuracy_score(ytest, prediction))
mosaic(conf_matrix)


# ---
# 
# ## Final Reduced Multiple Regression Model
# 
# At this point, I have eliminated any sources of multicollinearity and collinearity as well as variables exhibiting p-values that exceed .05.  I will finalize the reduced model and check to see how it compares to my initial model which included all variables in the set.

# In[ ]:


# Create final reduced model and display summary
mdl_churn_vs_features_final = logit("Churn_cat ~ MonthlyCharge + Tenure",
                           data=df_data).fit()
print(mdl_churn_vs_features_final.summary())


# In[ ]:


# Display confusion matrix, accuracy score, and mosaic for initial model and final reduced model for comparison
Xorig = df_data[['Area_cat', 'Children', 'Age', 'Income', 'Marital_cat', 'Gender_cat', 'Outage_sec_perweek', 
           'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year']]
yorig = df_data['Churn_cat']
yhat_orig = mdl_churn_vs_all.predict(Xorig)
prediction_orig = list(map(round, yhat_orig))
conf_matrix_orig = confusion_matrix(yorig, prediction_orig)
Xfinal = df_data[['MonthlyCharge', 'Tenure']]
yfinal = df_data['Churn_cat']
yhat_final = mdl_churn_vs_features_final.predict(Xfinal)
prediction_final = list(map(round, yhat_final))
conf_matrix_final = confusion_matrix(yfinal, prediction_final)
print ("Orignal Confusion Matrix : \n", conf_matrix_orig)
print('Orignal accuracy = ', accuracy_score(yorig, prediction_orig))
print ("Final Confusion Matrix : \n", conf_matrix_final)
print('Final accuracy = ', accuracy_score(yfinal, prediction_final))
mosaic(conf_matrix_orig)
mosaic(conf_matrix_final)


# ---
# 
# The reduced model holds up well when compared to the initial model using the accuracy score and confusion matrix as my measurements.
# 
# I will perform due diligence and check for extreme outliers to make sure my independent variables comply with the assumptions of logistic regression.

# In[ ]:


sns.boxplot(x='MonthlyCharge',data=df_data)


# In[ ]:


sns.boxplot(x='Tenure',data=df_data)


# ---
# 
# Neither variable has extreme outliers.
# 
# While Python does not have a library supporting the Box-Tidwell test with a single line of code, I can use a combination of functions from the statsmodels package to perform the test.  The goal will be to verify that the "MonthlyCharge:Log_MonthlyCharge" and "Tenure:Log_Tenure" interactions have p-values greater than 0.05, implying that both independent variables are linearly related to the logit of the dependent variable, Churn_cat.

# In[ ]:


# Define continuous variables
continuous_var = ['MonthlyCharge', 'Tenure']

# Add logit transform interaction terms (natural log) for continuous variables e.g.. Age * Log(Age)
for var in continuous_var:
    df_data[f'{var}:Log_{var}'] = df_data[var].apply(lambda x: x * np.log(x))

# Keep columns related to continuous variables
cols_to_keep = continuous_var + df_data.columns.tolist()[-len(continuous_var):]

# Redefining variables to include interaction terms
X_lt = df_data[cols_to_keep]
y_lt = df_data['Churn_cat']

# Add constant term
X_lt_constant = sm.add_constant(X_lt, prepend=False)
  
# Building model and fit the data (using statsmodel's Logit)
logit_results = GLM(y_lt, X_lt_constant, family=families.Binomial()).fit()

# Display summary results
print(logit_results.summary())


# ---
# 
# ## Data Analysis Process
# 
# During my variable selection process, I relied upon trusted methods for identifying variables unsuitable for the model, such as VIF, a correlation table, and p-values.  I measured each model's performance by its accuracy score, as well as the confusion matrix.

# ---
# 
# # **Part V: Data Summary and Implications**
# 
# ## Summary of Findings
# 
# The regression equation for the final reduced model is as follows:
# 
# **Churn_cat ~ MonthlyCharge + Tenure**
# 
# The coefficients for each variable included:
# 
# **MonthlyCharge     0.0332**
# 
# **Tenure           -0.0738**
# 
# We can use these coefficients to determine the effect each variable will have on a customer's decision to cancel service.  MonthlyCharge has a positive coefficient, indicating the higher a customer's monthly charge is, the more likely they are to churn.  Contrarily, Tenure has a negative coefficient, which tells us that customers who retain service for longer periods of time grow less and less likely to churn.
# 
# The model can provide significant data when evaluating customer retention from a practical perspective, as customers who have been with the company for a long time may be less likely to cancel service, but if the rate they are charged increases so do the chances they will churn.  The limitations of using logistic regression models for practical purposes are always present, however.  They are susceptible to overfitting and can appear to have more predictive power than they do (Robinson, 2018).  Also, as logistic regressions cannot predict continuous outcomes, their predictions contain less detail.  For instance, this model may be able to predict what makes a customer more likely to churn, but not when they might do so.
# 
# ---
# 
# ## Recommended Course of Action
# 
# There are a few key takeaways based on the analysis of this model.  For each month the customer stays with the ISP they are less likely to churn, but if their monthly fee increases they are more likely to churn.  It may be beneficial to offer long-standing customers occasional discounted rates to further increase the chance they will retain service.  For newer customers who already run a higher risk of churning, increasing their rates should be avoided altogether.

# ---
# 
# # **Part VI: Demonstration**
# 
# **Panopto Video Recording**
# 
# A link for the Panopto video has been provided separately.  The demonstration includes the following:
# 
# •  Demonstration of the functionality of the code used for the analysis
# 
# •  Identification of the version of the programming environment
# 
# •  Comparison of the two multiple regression models you used in your analysis
# 
# •  Interpretation of the coefficients
# 

# ---
# 
# # **Web Sources**
# 
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html
# 
# https://pbpython.com/categorical-encoding.html
# 
# https://towardsdatascience.com/assumptions-of-logistic-regression-clearly-explained-44d85a22b290
# 
# 

# ---
# 
# # **References**
# 
# 
# Insights for Professionals. (2019, February 26). *5 Niche Programming Languages (And Why They're Underrated).* https://www.insightsforprofessionals.com/it/software/niche-programming-languages
# 
# 
# Parra, H.  (2021, April 20).  *The Data Science Trilogy.*  Towards Data Science.  https://towardsdatascience.com/the-data-science-trilogy-numpy-pandas-and-matplotlib-basics-42192b89e26
# 
# 
# Zach.  (2021, November 16).  *The 6 Assumptions of Logistic Regression (With Examples).*  Statology.  https://www.statology.org/assumptions-of-logistic-regression/
# 
# 
# Robinson, N.  (2018, June 28).  *The Disadvantages of Logistic Regression.*  The Classroom.  https://www.theclassroom.com/multivariate-statistical-analysis-2448.html
# 
