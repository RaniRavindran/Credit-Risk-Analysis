#!/usr/bin/env python
# coding: utf-8

# In[127]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[128]:


import seaborn as sns
# Pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None, 'display.max_columns', None)

# Plotting options
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set(style='whitegrid')


# In[129]:


loans = pd.read_csv('accepted_2007_to_2018.csv')


# In[77]:


loans.shape


# In[5]:


loans['loan_status'].value_counts(dropna=False)


# In[130]:


loans = loans.loc[loans['loan_status'].isin(['Fully Paid', 'Charged Off'])]


# In[131]:


loans.shape


# In[8]:


loans['loan_status'].value_counts(dropna=False)


# In[9]:


#loan_status in percentage
loans['loan_status'].value_counts(normalize=True, dropna=False)


# In[132]:


#drop features that are missing 30% of missing data
#for that first calculate the missing % of each feature
missing_fractions = loans.isnull().mean().sort_values(ascending=False)


# In[400]:


missing_fractions.sample(20)


# In[244]:


plt.figure(figsize=(6,3), dpi=90)
missing_fractions.plot.hist(bins=20)
plt.title('Histogram of Feature Incompleteness')
plt.xlabel('Fraction of data missing')
plt.ylabel('Feature count')


# In[133]:


#store all variables missing more than 30% data in an alphabetical list:
drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
#print(drop_list)


# In[12]:


#number of features that will be dropped
len(drop_list)


# In[134]:


#dropping the features
loans.drop(labels=drop_list, axis=1, inplace=True)


# In[13]:


loans.shape


# In[135]:


#features that has to be kept is determined from the research papers and data dictionary provided
keep_list = ['annual_inc', 'application_type', 'dti', 'delinq_2yrs','earliest_cr_line', 'emp_length','emp_title', 'fico_range_high', 
             'fico_range_low', 'grade', 'home_ownership', 'initial_list_status', 'installment', 'int_rate','inq_last_6mths',
             'loan_amnt', 'loan_status', 'mort_acc', 'mths_since_last_delinq','open_acc', 'pub_rec','pub_rec_bankruptcies',
             'purpose', 'revol_bal','revol_util','sub_grade', 'term', 'title', 'total_acc','verification_status','issue_d']


# In[81]:


len(keep_list)


# In[136]:


drop_list = [col for col in loans.columns if col not in keep_list]
#print(drop_list)


# In[17]:


len(drop_list)


# In[137]:


#drop the unwanted fetures
loans.drop(labels=drop_list, axis=1, inplace=True)


# In[84]:


loans.shape


# In[73]:


loans.dtypes


# In[138]:


#PREPROVESSING 
#Drop the feature if it is not useful for predicting the loan status.
#View summary statistics and visualize the data, plotting against the loan status.
#Modify the feature to make it useful for modeling, if necessary.
import seaborn as sns
def plot_var(col_name, full_name, continuous):
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    
    # Plot without loan status
    if continuous:
        sns.distplot(loans.loc[loans[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(loans[col_name], order=sorted(loans[col_name].unique()), color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(full_name)
    ax1.set_ylabel('Count')
    ax1.set_title(full_name)

    # Plot with loan status
    if continuous:
        sns.boxplot(x=col_name, y='loan_status', data=loans, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(full_name + ' by Loan Status')
    else:
        charge_off_rates = loans.groupby(col_name)['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
        sns.barplot(x=charge_off_rates.index, y=charge_off_rates.values, color='#5975A4', saturation=1, ax=ax2)
        ax2.set_ylabel('Fraction of Loans Charged-off')
        ax2.set_title('Charge-off Rate by ' + full_name)
    ax2.set_xlabel(full_name)
    
    plt.tight_layout()


# # loan amount

# In[525]:


#LOAN AMOUNT
loans['loan_amnt'].describe()


# In[254]:


plot_var('loan_amnt', 'Loan Amount', continuous=True)


# In[255]:


loans['loan_amnt'].isnull().any()


# There is no null value in'loan_amt'

# In[858]:


loans.groupby('loan_status')['loan_amnt'].describe()


# # Term
# 

# In[859]:


loans['term'].value_counts(dropna=False)


# convert the feature 'Term' to integer

# In[139]:


loans['term'] = loans['term'].apply(lambda s: np.int8(s.split()[0]))


# In[38]:


loans['term'].value_counts(normalize=True)


# In[862]:


loans['term'].isnull().any()


# There is no null value in 'term'

# In[257]:


plot_var('term', 'TERM', continuous=False)


# In[71]:


loans.groupby('term')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']


# In[70]:


loans.groupby('loan_status')['term'].describe()


# # int_rate ....interest rate of loan

# In[21]:


loans['int_rate'].describe()


# In[864]:


plot_var('int_rate', 'Interest Rate', continuous=True)


# In[865]:


loans.groupby('loan_status')['int_rate'].describe()


# In[866]:


loans['int_rate'].isnull().any()


# # instalment

# In[867]:


loans['installment'].isnull().any()


# In[868]:


loans['installment'].describe()


# ###### Installments range from $4.93 to $1,714, with a median of $377

# In[668]:


plot_var('installment', 'Installment', continuous=True)


# In[869]:


loans.groupby('loan_status')['installment'].describe()


# # GRADE AND SUB_GRADE

# In[870]:


print(sorted(loans['grade'].unique()))


# In[871]:


print(sorted(loans['sub_grade'].unique()))


# ###### The grade is implied by the subgrade, so let's drop the grade column.

# In[140]:


loans.drop('grade', axis=1, inplace=True)


# In[497]:


plot_var('sub_grade', 'Subgrade', continuous=False)


# In[873]:


loans['sub_grade'].isnull().any()


# ###### There's a clear trend of higher probability of charge-off as the subgrade worsens.

# # emp_title

# In[874]:


loans['emp_title'].describe()


# ###### too many titles so drop it

# In[141]:


loans.drop(labels='emp_title', axis=1, inplace=True)


# # emp_length

# In[876]:


loans['emp_length'].value_counts(dropna=False).sort_index()


# ###### Note there are 42,253 loans without data on the length of employment.
# 
# Convert emp_length to integers:

# In[142]:


loans['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)


# In[143]:


loans['emp_length'].replace('< 1 year', '0 years', inplace=True)


# In[144]:


def emp_length_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])


# In[145]:


loans['emp_length'] = loans['emp_length'].apply(emp_length_to_int)


# In[146]:


loans['emp_length'].value_counts(dropna=False).sort_index()


# In[882]:


loans.dtypes


# ###### Loan status does not appear to vary much with employment length on average, except for a small drop in charge-offs for borrowers with over 10 years of employment.

# In[48]:


plot_var('emp_length', 'Employment Length', continuous=False)


# In[168]:


loans[loans['emp_length'].isnull()]
print('median:', loans['emp_length'].median())


# In[148]:


loans['emp_length'].fillna(loans['emp_length'].median(),inplace=True)


# In[149]:


loans['emp_length'].isnull().any()


# # home_ownership

# ###### Data Dictionary: "The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER."

# In[887]:


loans['home_ownership'].value_counts(dropna=False)


# ###### Replace the values ANY and NONE with OTHER:

# In[150]:


loans['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)


# In[29]:


loans['home_ownership'].value_counts(dropna=False)


# In[890]:


plot_var('home_ownership', 'Home Ownership', continuous=False)


# In[30]:


loans.groupby('home_ownership')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']


# In[892]:


loans['home_ownership'].isnull().any()


# # annual_inc

# In[893]:


loans['annual_inc'].describe()


# Annual income ranges from $0 to $9,550,000, with a median of $65,000. Because of the large range of incomes, we should take a log transform of the annual income variable.

# In[151]:


loans['log_annual_inc'] = loans['annual_inc'].apply(lambda x: np.log10(x+1))


# In[152]:


loans.drop('annual_inc', axis=1, inplace=True)


# In[419]:


loans['log_annual_inc'].describe()


# In[363]:


plot_var('log_annual_inc', 'Log Annual Income', continuous=True)


# It appears that individuals with higher income are more likely to pay off their loans

# In[897]:


#summart statics
loans.groupby('loan_status')['log_annual_inc'].describe()


# In[898]:


loans['log_annual_inc'].isnull().any()


# # verification_status

# In[55]:


loans['verification_status'].value_counts(dropna=False)


# In[697]:


plot_var('verification_status', 'Verification Status', continuous=False)


# In[153]:


loans['verification_status'].replace(['Verified'], 'Source Verified', inplace=True)


# In[482]:


loans['verification_status'].value_counts(dropna=False)


# In[900]:


plot_var('verification_status', 'Verification Status', continuous=False)


# In[901]:


loans['verification_status'].isnull().any()


# # issue_d

# Data Dictionary: "The month which the loan was funded."
# 
# Because we're only using variables available to investors before the loan was funded, issue_d will not be included in the final model. We're keeping it for now just to perform the train/test split later, then we'll drop it.

# In[154]:


loans.drop('issue_d', axis=1, inplace=True)


# # purpose

# In[59]:


loans['purpose'].value_counts(dropna=False)


# In[35]:


loans.groupby('purpose')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off'].sort_values()


# In[272]:


plot_var('purpose', 'purpose of loan', continuous=False)


# Notice that only 12% of completed loans for weddings have charged-off, but 30% of completed small business loans have charged-off.

# In[905]:


loans['purpose'].isnull().any()


# # title

# In[906]:


loans['title'].describe()


# In[907]:


#top 10 title and their frequencies
loans['title'].value_counts().head(10)


# There are 60,298 different titles in the dataset, and based on the top 10 titles, the purpose variable appears to already contain this information. So we drop the title variable

# In[155]:


loans.drop('title', axis=1, inplace=True)


# # zipcode and addr_state is already dropped

# # dti

# "A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income."

# In[61]:


loans['dti'].describe()


# In[37]:


loans[loans['dti'].isnull()]
print('median:', loans['dti'].median())


# In[156]:


loans['dti'].fillna(loans['dti'].median(),inplace=True)


# In[67]:


loans['dti'].describe()


# In[64]:


plt.figure(figsize=(8,3), dpi=90)
sns.distplot(loans.loc[loans['dti'].notnull() & (loans['dti']<115) &(loans['dti']>0), 'dti'], kde=False)
plt.xlabel('Debt-to-income Ratio')
plt.ylabel('Count')
plt.title('Debt-to-income Ratio')


# In[65]:


(loans['dti']>=100).sum()


# In[912]:


(loans['dti']<0).sum()


# In[66]:


loans.groupby('loan_status')['dti'].describe()


# In[914]:


#loans['dti'].isnull().any()


# In[179]:


#loans[loans['dti'].isnull()]
##print('median:', loans['dti'].median())


# In[180]:


#loans['dti'].fillna(loans['dti'].median(),inplace=True)


# In[275]:


loans['dti'].isnull().any()


# In[68]:


plot_var('dti', 'Debt to Income ratio', continuous=True)


# # earliest_cr_line

#  "The month the borrower's earliest reported credit line was opened."

# In[104]:


loans['earliest_cr_line'].sample(5)


# In[105]:


loans['earliest_cr_line'].isnull().any()


# In[106]:


loans['earliest_cr_line'] = loans['earliest_cr_line'].apply(lambda s: int(s[-2:]))


# In[70]:


loans['earliest_cr_line'].describe()


# In[389]:


plot_var('earliest_cr_line', 'Year of Earliest Credit Line', continuous=True)


# In[922]:


#loans.drop('earliest_cr_line', axis=1, inplace=True)
loans['earliest_cr_line'].dtypes


# In[157]:


############New modification#################################


loans.drop('earliest_cr_line', axis=1, inplace=True)


# # fico_range_low, fico_range_high

# In[924]:


loans[['fico_range_low', 'fico_range_high']].describe()


# Check the Pearson correlation between these values:

# In[925]:


loans[['fico_range_low','fico_range_high']].corr()


# We only need to keep one of the FICO scores. We'll take the average of the two and call it fico_score:

# In[158]:


loans['fico_score'] = 0.5*loans['fico_range_low'] + 0.5*loans['fico_range_high']


# In[159]:


loans.drop(['fico_range_high', 'fico_range_low'], axis=1, inplace=True)


# In[106]:


plot_var('fico_score', 'FICO Score', continuous=True)


# There is a noticeable difference in FICO scores between fully paid and charged-off loans. Compare the summary statistics:

# In[928]:


loans.groupby('loan_status')['fico_score'].describe()


# Loans that charge off have a FICO score 10 points lower on average.

# In[929]:


loans['fico_score'].describe()


# In[930]:


loans['fico_score'].isnull().any()


# # open_acc

# Data Dictionary: "The number of open credit lines in the borrower's credit file."

# In[110]:


plt.figure(figsize=(10,3), dpi=90)
sns.countplot(loans['open_acc'], order=sorted(loans['open_acc'].unique()), color='#5975A4', saturation=1)
_, _ = plt.xticks(np.arange(0, 90, 5), np.arange(0, 90, 5))
plt.title('Number of Open Credit Lines')


# In[931]:


plot_var('open_acc', 'Open credit lines', continuous=True)


# In[932]:


loans.groupby('loan_status')['open_acc'].describe()


# In[933]:


loans['open_acc'].describe()


# In[934]:


loans['open_acc'].isnull().any()


# # pub_rec

# Data Dictionary: "Number of derogatory public records."

# In[935]:


loans['pub_rec'].value_counts().sample(5)


# In[936]:


loans['pub_rec'].value_counts().sort_index()


# Is there a difference in average public records between fully paid loans and charged-off loans?

# In[937]:


loans.groupby('loan_status')['pub_rec'].describe()


# In[938]:


loans['pub_rec'].isnull().any()


# In[527]:


plot_var('pub_rec', 'Public Drogatory Records', continuous=True)


# # revol_bal

# Data Dictionary: "Total credit revolving balance."

# In[939]:


loans['revol_bal'].describe()


# In[160]:


loans['log_revol_bal'] = loans['revol_bal'].apply(lambda x: np.log10(x+1))


# In[43]:


plot_var('log_revol_bal', 'Log Revolving Credit Balance', continuous=True)


# In[941]:


loans.groupby('loan_status')['log_revol_bal'].describe()


# In[161]:


loans.drop('revol_bal', axis=1, inplace=True)


# There is no much difference in the mean. So better to drop

# In[280]:


loans['log_revol_bal'].isnull().any()


# # revol_util

# Data Dictionary: "Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit."

# In[944]:


loans['revol_util'].describe()


# In[528]:


loans['revol_util'].value_counts().sort_index()


# In[945]:


plot_var('revol_util', 'Revolving Line Utilization', continuous=True)


# In[946]:


loans.groupby('loan_status')['revol_util'].describe()


# In[947]:


#loans.drop('revol_util', axis=1, inplace=True)
loans['revol_util'].isnull().any()


# In[281]:


print('median:', loans['revol_util'].median())


# In[162]:


loans['revol_util'].fillna(loans['revol_util'].median(), inplace=True)


# In[163]:


loans['revol_util'].isnull().any()


# In[491]:


#loans.drop('revol_util', axis=1, inplace=True)


# # total_acc

#  "The total number of credit lines currently in the borrower's credit file."

# In[131]:


plt.figure(figsize=(12,3), dpi=90)
sns.countplot(loans['total_acc'], order=sorted(loans['total_acc'].unique()), color='#5975A4', saturation=1)
_, _ = plt.xticks(np.arange(0, 176, 10), np.arange(0, 176, 10))
plt.title('Total Number of Credit Lines')


# In[951]:


loans.groupby('loan_status')['total_acc'].describe()


# #no large difference so may consider to drop

# In[952]:


loans['total_acc'].isnull().any()


# In[164]:


loans.drop('total_acc', axis=1, inplace=True)


# # initial_list_status

# Data Dictionary: "The initial listing status of the loan. Possible values are – W, F." I'm not sure what this means

# In[134]:


plot_var('initial_list_status', 'Initial List Status', continuous=False)


# In[953]:


loans['initial_list_status'].isnull().any()


# donot know whatit is . So bettter to drop

# In[165]:


loans.drop('initial_list_status', axis=1, inplace=True)


# # application_type

# Data Dictionary: "Indicates whether the loan is an individual application or a joint application with two co-borrowers."

# In[955]:


loans['application_type'].value_counts()


# In[956]:


loans.groupby('application_type')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']


# no significant difference to be charged-off.

# In[957]:


loans['application_type'].isnull().any()


# In[494]:


#loans.drop('application_type', axis=1, inplace=True)
plot_var('application_type', 'Application Type', continuous=False)


# # mort_acc

# Data Dictionary: "Number of mortgage accounts."

# In[958]:


loans['mort_acc'].describe()


# Not sure how someone can have 51 mortgage accounts...but apparently they do. Check the top 10 values:

# In[53]:


loans['mort_acc'].value_counts().head(20)


# In[960]:


loans.groupby('loan_status')['mort_acc'].describe()


# In[961]:


loans['mort_acc'].isnull().any()


# In[166]:


#loans[loans['mort_acc'].isnull()]
#print('median:', loans['mort_acc'].median())
loans['mort_acc'].fillna(loans['mort_acc'].median(),inplace=True)


# Individuals who pay off their loans are more likely to have several mortgage account

# In[167]:


loans['mort_acc'].isnull().any()


# # pub_rec_bankruptcies

# Data Dictionary: "Number of public record bankruptcies."

# In[964]:


loans['pub_rec_bankruptcies'].value_counts().sort_index()


# In[965]:


plot_var('pub_rec_bankruptcies', 'Public Record Bankruptcies', continuous=False)


# In[966]:


loans['pub_rec_bankruptcies'].isnull().any()


# In[168]:





#loans[loans['pub_rec_bankruptcies'].isnull()]
#print('median:', loans['pub_rec_bankruptcies'].median())
loans['pub_rec_bankruptcies'].fillna(loans['pub_rec_bankruptcies'].median(),inplace=True)


# In[169]:


loans['pub_rec_bankruptcies'].isnull().any()


# # delinq_2yrs

# In[969]:


loans['delinq_2yrs'].describe()


# In[970]:


loans['delinq_2yrs'].isnull().any()


# In[170]:


loans.drop('delinq_2yrs', axis=1, inplace=True)


# # inq_last_6mts

# In[971]:


loans['inq_last_6mths'].describe()


# In[972]:


loans['inq_last_6mths'].isnull().any()


# In[192]:


loans[loans['inq_last_6mths'].isnull()]
print('median:', loans['inq_last_6mths'].median())


# In[51]:


loans['inq_last_6mths'].fillna(loans['inq_last_6mths'].median(), inplace=True)


# In[975]:


loans['inq_last_6mths'].isnull().any()


# In[171]:


loans.drop('inq_last_6mths', axis=1, inplace=True)


# # loan_status

# In[976]:


loans['loan_status'].isnull().any()


# In[172]:


loans.shape


# In[54]:


#loan_status in percentage
loans['loan_status'].value_counts(normalize=True, dropna=False)


# # More Pre-processing

# Before that we need to convert our categorical variables level into numbers
# For that we will do One Hot Encoding

# In[502]:


loans.shape


# In[434]:


loans.dtypes


# In[173]:


from sklearn.preprocessing import LabelEncoder


# In[174]:


# Here the column loan_satus is convered to o and 1. o for 'Fully Paid'and 1 for 'Charged Off'. 
#The column replaced to charged_off
    
loans['charged_off'] = (loans['loan_status'] == 'Charged Off').apply(np.uint8)


# In[983]:


loans.head(10)


# In[175]:


loans.drop('loan_status',axis=1, inplace=True)


# In[103]:


loans.dtypes


# In[56]:


loans['charged_off'].unique()


# In[439]:


loans['charged_off'].value_counts()


# In[54]:


#loan_status in percentage
loans['charged_off'].value_counts(normalize=True, dropna=False)


# In[176]:


loans = pd.get_dummies(loans, columns=['sub_grade', 'home_ownership', 'verification_status', 
                                       'purpose', 'application_type'], drop_first=True)


# # Splitting the data frame to train and test 

# In[177]:


# split the data into training and testing
from sklearn.model_selection import train_test_split
X = loans.ix[:, loans.columns != "charged_off"]
y = loans["charged_off"]

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.3, random_state=44)


# In[178]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[124]:


X_train.shape


# # Applying Algorithms

# We can clearly see that due to dominace of one class that is Good Loans all are alogrithms are predicting good loans with high accuracy but Bad Loans with a very low accuracy
# There is class imbalance problem
# We have to use SMOTE package and try over sampling and undersampling to see if there is any improvement

# In[179]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report


# # logistic Regression

# In[63]:


# Logistic Regression
log= LogisticRegression()
log.fit(X_train, y_train)

y_pred= log.predict(X_test)

# Summary of the prediction
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy
print("Accuracy of the model is: ", accuracy_score(y_pred,y_test))


# In[194]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve Logistic Regression for class 1/Charged off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs = log.predict_proba(X_test)
probs = probs[:,0]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)


# In[197]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression_class 0/fully paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs = log.predict_proba(X_test)
probs = probs[:,1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)


# In[126]:


print(log.get_params())


# # Logistic Regression Undersampling

# In[64]:


from imblearn.under_sampling import RandomUnderSampler

print("Before undersampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before underSampling, counts of label '0': {} \n".format(sum(y_train==0)))

# Under Sampling
rus = RandomUnderSampler(random_state=0)
X_train_res, y_train_res = rus.fit_sample(X_train, y_train.ravel())

print("After UnderSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After UnderSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[65]:


# Logistic Regression with oversampling
log_un= LogisticRegression()
log_un.fit(X_train_res, y_train_res)

y_pred_un= log_un.predict(X_test)

# Summary of the prediction
print(classification_report(y_test, y_pred_un))
print(confusion_matrix(y_test, y_pred_un))

# Accuracy
print("Accuracy of the model is: ", accuracy_score(y_pred_un,y_test))


# In[66]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve of Logistic Regression undersampling class 1/charged off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_un_log = log_un.predict_proba(X_test)
probs_un_log = probs_un_log[:,0]
auc = roc_auc_score(y_test, probs_un_log)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_un_log)
plot_roc_curve(fpr, tpr)


# In[68]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve of Logistic Regression undersampling class 0/fully paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_un_log = log_un.predict_proba(X_test)
probs_un_log = probs_un_log[:,1]
auc = roc_auc_score(y_test, probs_un_log)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_un_log)
plot_roc_curve(fpr, tpr)


# # oversampling of Logistic Regression

# In[75]:


# from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

# Over Sampling
ros = RandomOverSampler(random_state=0)
X_train_over, y_train_over = ros.fit_sample(X_train, y_train.ravel())

print("After OverSampling, counts of label '1': {}".format(sum(y_train_over==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_over==0)))


# In[76]:


# Logistic Regression with oversampling
log_ov= LogisticRegression()
log_ov.fit(X_train_over, y_train_over)

y_pred_ov= log_ov.predict(X_test)

# Summary of the prediction
print(classification_report(y_test, y_pred_ov))
print(confusion_matrix(y_test, y_pred_ov))

# Accuracy
print("Accuracy of the model is: ", accuracy_score(y_pred_ov,y_test))


# In[73]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve oversampling Logistic Regression class 1/charged off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_ov_log = log_ov.predict_proba(X_test)
probs_ov_log = probs_ov_log[:,0]
auc = roc_auc_score(y_test, probs_ov_log)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_ov_log)
plot_roc_curve(fpr, tpr)


# In[74]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve oversampling Logistic Regression class o/Fully Paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_ov_log = log_ov.predict_proba(X_test)
probs_ov_log = probs_ov_log[:,1]
auc = roc_auc_score(y_test, probs_ov_log)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_ov_log)
plot_roc_curve(fpr, tpr)


# In[115]:


pd.DataFrame(data={'predictions': y_pred, 'actual': y_test})


# # Prediction for train data for logistic Regression

# In[352]:


y1_pred=log.predict(X_train)
#summary of prediction of training data set

print(confusion_matrix(y_train, y1_pred))

# Accuracy
print("Accuracy of the model is: ", accuracy_score(y1_pred,y_train))


# # KNN

# In[1023]:


from sklearn.neighbors import NearestNeighbors


# In[1024]:


neigh = NearestNeighbors()


# In[1025]:


neigh.fit(X_train,y_train)


# In[1026]:


y_pred=neigh.predict(X_test)


# # Naive Bayes

# In[180]:



# Naive Bayes
naive= GaussianNB()
naive.fit(X_train, y_train)

y_pred_naive= naive.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred_naive))
print(confusion_matrix(y_test, y_pred_naive))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_naive,y_test))


# In[181]:


print(naive.get_params())


# In[78]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Naive Bayesian for class 1/Charged Off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_naive = naive.predict_proba(X_test)
probs_naive = probs_naive[:,0]
auc = roc_auc_score(y_test, probs_naive)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_naive)
plot_roc_curve(fpr, tpr)


# In[79]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Naive Bayesian for class 0/Fully Paid ')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_naive = naive.predict_proba(X_test)
probs_naive = probs_naive[:,1]
auc = roc_auc_score(y_test, probs_naive)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_naive)
plot_roc_curve(fpr, tpr)


# # Over sampling Naive Bayesian

# In[80]:


# Naive Bayes with over sampling
naive_ov= GaussianNB()
naive_ov.fit(X_train_over, y_train_over)

y_pred_naive_ov= naive_ov.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred_naive_ov))
print(confusion_matrix(y_test, y_pred_naive_ov))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_naive_ov,y_test))


# In[81]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve after oversampling Naive Bayesian For Class 1/charged off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_naive_ov = naive_ov.predict_proba(X_test)
probs_naive_ov = probs_naive_ov[:,0]
auc = roc_auc_score(y_test, probs_naive_ov)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_naive_ov)
plot_roc_curve(fpr, tpr)


# In[82]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve after over sampling for Naive Bayesian for class 0/Fully paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_naive_ov = naive_ov.predict_proba(X_test)
probs_naive_ov = probs_naive_ov[:,1]
auc = roc_auc_score(y_test, probs_naive_ov)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_naive_ov)
plot_roc_curve(fpr, tpr)


# # Naive Bayesian undersampling 

# In[83]:


# Naive Bayes with over sampling
naive_un= GaussianNB()
naive_un.fit(X_train_res, y_train_res)

y_pred_naive_un= naive_un.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred_naive_un))
print(confusion_matrix(y_test, y_pred_naive_un))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_naive_un,y_test))


# In[84]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve after under sampling for Naive Bayesian for class 0/Fully paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_naive_un = naive_un.predict_proba(X_test)
probs_naive_un = probs_naive_un[:,1]
auc = roc_auc_score(y_test, probs_naive_un)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_naive_un)
plot_roc_curve(fpr, tpr)


# In[85]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve after under sampling for Naive Bayesian for class 1/charged off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_naive_un = naive_un.predict_proba(X_test)
probs_naive_un = probs_naive_un[:,0]
auc = roc_auc_score(y_test, probs_naive_un)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_naive_un)
plot_roc_curve(fpr, tpr)


# In[1007]:


type(X_train)


# In[1008]:


type(y_test)


# In[1009]:


type(y_train)


# In[124]:


pd.DataFrame(data={'predictions': y_pred_naive, 'actual': y_test})


# # Random Forest

# In[86]:


from sklearn.ensemble import RandomForestClassifier


# In[87]:



rf= RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf= rf.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
accuracy=accuracy_score(y_pred_rf,y_test)
# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_rf,y_test))


# In[88]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest for class 1/Charged Off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_rf = rf.predict_proba(X_test)
probs_rf = probs_rf[:,0]
auc = roc_auc_score(y_test, probs_rf)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_rf)
plot_roc_curve(fpr, tpr)


# In[89]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest for class 0/Fully Paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_rf = rf.predict_proba(X_test)
probs_rf = probs_rf[:,1]
auc = roc_auc_score(y_test, probs_rf)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_rf)
plot_roc_curve(fpr, tpr)


# # Random Forest Oversampling

# In[90]:


##oversamplinf Random Forest
rf_ov= RandomForestClassifier()
rf_ov.fit(X_train_over, y_train_over)

y_pred_rf_ov= rf_ov.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred_rf_ov))
print(confusion_matrix(y_test, y_pred_rf_ov))
auc_rf=accuracy_score(y_pred_rf_ov,y_test)
# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_rf_ov,y_test))


# In[139]:


print(rf.get_params())


# In[91]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc_rf)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve oversampling Random Forest class 1/charged off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_rf_ov = rf_ov.predict_proba(X_test)
probs_rf_ov = probs_rf_ov[:,0]
auc_rf = roc_auc_score(y_test, probs_rf_ov)
print('AUC: %.2f' % auc_rf)
fpr, tpr, thresholds = roc_curve(y_test, probs_rf_ov)
plot_roc_curve(fpr, tpr)


# In[92]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve over sampling class 0/fully paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_naive_ov = naive_ov.predict_proba(X_test)
probs_naive_ov = probs_naive_ov[:,1]
auc = roc_auc_score(y_test, probs_naive_ov)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_naive_ov)
plot_roc_curve(fpr, tpr)


# # Under sampling Random Forest

# In[93]:


##oversamplinf Random Forest
rf_un= RandomForestClassifier()
rf_un.fit(X_train_res, y_train_res)

y_pred_rf_un= rf_un.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred_rf_un))
print(confusion_matrix(y_test, y_pred_rf_un))
auc_rf=accuracy_score(y_pred_rf_un,y_test)
# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_rf_un,y_test))


# In[94]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc_rf)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve undersampling Random Forest class 1/charged off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_rf_un = rf_un.predict_proba(X_test)
probs_rf_un = probs_rf_un[:,0]
auc_rf = roc_auc_score(y_test, probs_rf_un)
print('AUC: %.2f' % auc_rf)
fpr, tpr, thresholds = roc_curve(y_test, probs_rf_un)
plot_roc_curve(fpr, tpr)


# In[96]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc_rf)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve undersampling Random Forest Class 0/Fully Paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_rf_un = rf_un.predict_proba(X_test)
probs_rf_un = probs_rf_un[:,1]
auc_rf = roc_auc_score(y_test, probs_rf_un)
print('AUC: %.2f' % auc_rf)
fpr, tpr, thresholds = roc_curve(y_test, probs_rf_un)
plot_roc_curve(fpr, tpr)


# # xgboost classification

# In[798]:


#pip install --user xgboost


# In[97]:


from xgboost.sklearn import XGBClassifier


# In[98]:



# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)



# In[99]:



# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[100]:


print(confusion_matrix(y_test, y_pred))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred,y_test))


# In[101]:


# Summary of prediction
print(classification_report(y_test, y_pred))


# In[102]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for XgBoost class 1/charged off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_xg = model.predict_proba(X_test)
probs_xg = probs_xg[:,0]
auc = roc_auc_score(y_test, probs_xg)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_xg)
plot_roc_curve(fpr, tpr)


# In[103]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve XgBoost class 0/Fuly Paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_xg = model.predict_proba(X_test)
probs_xg = probs_xg[:,1]
auc = roc_auc_score(y_test, probs_xg)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_xg)
plot_roc_curve(fpr, tpr)


# # XgBoost OverSampling

# In[104]:


##oversamplinf Xg boost
xg_ov= XGBClassifier()
xg_ov.fit(X_train_over, y_train_over)

y_pred_xg_ov= xg_ov.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred_xg_ov))
print(confusion_matrix(y_test, y_pred_xg_ov))
auc_rf=accuracy_score(y_pred_xg_ov,y_test)
# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_xg_ov,y_test))


# In[105]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve  Oversampling XgBoost class 0/Fuly Paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_xg_ov = xg_ov.predict_proba(X_test)
probs_xg_ov = probs_xg_ov[:,1]
auc = roc_auc_score(y_test, probs_xg_ov)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_xg_ov)
plot_roc_curve(fpr, tpr)


# In[106]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve  Oversampling XgBoost class 1/charged Off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_xg_ov = xg_ov.predict_proba(X_test)
probs_xg_ov = probs_xg_ov[:,0]
auc = roc_auc_score(y_test, probs_xg_ov)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_xg_ov)
plot_roc_curve(fpr, tpr)


# # Xg Boost Undersampling

# In[107]:


##under Xg boost
xg_un= XGBClassifier()
xg_un.fit(X_train_res, y_train_res)

y_pred_xg_un= xg_un.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred_xg_un))
print(confusion_matrix(y_test, y_pred_xg_un))
auc_rf=accuracy_score(y_pred_xg_un,y_test)
# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_xg_un,y_test))


# In[108]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve  underampling XgBoost class 1/charged Off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_xg_un = xg_un.predict_proba(X_test)
probs_xg_un = probs_xg_un[:,0]
auc = roc_auc_score(y_test, probs_xg_un)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_xg_un)
plot_roc_curve(fpr, tpr)


# In[109]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve  underampling XgBoost class 0/Fully Paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_xg_un = xg_un.predict_proba(X_test)
probs_xg_un = probs_xg_un[:,1]
auc = roc_auc_score(y_test, probs_xg_un)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_xg_un)
plot_roc_curve(fpr, tpr)


# # Neural network-Multi-layer perceptron

# In[110]:


from sklearn.neural_network import MLPClassifier


# In[111]:


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(5, 2), random_state=1)


# In[112]:


clf.fit(X_train, y_train)                         


# In[113]:


# make predictions for test data
y_pred_neu = clf.predict(X_test)
predictions = [round(value) for value in y_pred_neu]


# In[114]:


# Summary of prediction
print(classification_report(y_test, y_pred_neu))
print(confusion_matrix(y_test, y_pred_neu))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_neu,y_test))


# In[115]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve of Multilayer Perceptron for class 1/charged Off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_neu = clf.predict_proba(X_test)
probs_neu = probs_neu[:,0]
auc = roc_auc_score(y_test, probs_neu)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_neu)
plot_roc_curve(fpr, tpr)


# In[117]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Multilayer Perceptron Class 0/Fully Paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_neu = clf.predict_proba(X_test)
probs_neu = probs_neu[:,1]
auc = roc_auc_score(y_test, probs_neu)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_neu)
plot_roc_curve(fpr, tpr)


# # Neural NetWork Oversampling

# In[118]:


##oversamplinf Xg boost
clf_ov = MLPClassifier(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(5, 2), random_state=1)
clf_ov.fit(X_train_over, y_train_over)

y_pred_clf_ov= clf_ov.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred_clf_ov))
print(confusion_matrix(y_test, y_pred_clf_ov))
auc_rf=accuracy_score(y_pred_clf_ov,y_test)
# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_clf_ov,y_test))


# In[146]:



def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve of Oversampling Multilayer Perceptron for class 1/charged Off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_neu_ov = clf_ov.predict_proba(X_test)
probs_neu_ov = probs_neu_ov[:,0]
auc = roc_auc_score(y_test, probs_neu_ov)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_neu_ov)
plot_roc_curve(fpr, tpr)


# In[147]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve of Oversampling Multilayer Perceptron for class 0/Fully Paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_neu_ov = clf_ov.predict_proba(X_test)
probs_neu_ov = probs_neu_ov[:,1]
auc = roc_auc_score(y_test, probs_neu_ov)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_neu_ov)
plot_roc_curve(fpr, tpr)


# # Neural Network Undersampling

# In[148]:


##oversampling NN 
clf_un = MLPClassifier(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(5, 2), random_state=1)
clf_un.fit(X_train_res, y_train_res)

y_pred_clf_un= clf_un.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred_clf_un))
print(confusion_matrix(y_test, y_pred_clf_un))
auc_rf=accuracy_score(y_pred_clf_un,y_test)
# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_clf_un,y_test))


# In[149]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve of undersampling Multilayer Perceptron for class 1/charged Off')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_neu_un = clf_un.predict_proba(X_test)
probs_neu_un = probs_neu_un[:,0]
auc = roc_auc_score(y_test, probs_neu_un)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_neu_un)
plot_roc_curve(fpr, tpr)


# In[150]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve of Oversampling Multilayer Perceptron for class 0/fully paid')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_neu_un = clf_un.predict_proba(X_test)
probs_neu_un = probs_neu_un[:,1]
auc = roc_auc_score(y_test, probs_neu_un)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_neu_un)
plot_roc_curve(fpr, tpr)


# In[153]:


pd.DataFrame(data={'predictions': y_pred_neu, 'actual': y_test})


# # Ada boost classification

# In[171]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification


# In[176]:


X_train, y_train = make_classification(n_samples=1000, n_features=67,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)


# In[177]:


clf_ada = AdaBoostClassifier(n_estimators=100, random_state=1)


# In[178]:


clf_ada.fit(X_train, y_train)


# In[179]:


# make predictions for test data
y_pred_ada = clf_ada.predict(X_test)
predictions = [round(value) for value in y_pred_ada]


# In[105]:


# Summary of prediction
print(classification_report(y_test, y_pred_ada))
print(confusion_matrix(y_test, y_pred_ada))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_ada,y_test))


# # Ada boost SVM

# In[151]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification


# In[152]:


X_train, y_train = make_classification(n_samples=1000, n_features=67,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)


# In[153]:


from sklearn.svm import SVC


# In[154]:


from sklearn import metrics
svc=SVC(probability=True, kernel='linear')


# In[155]:


clf_ada_sv = AdaBoostClassifier(n_estimators=50, base_estimator=svc, learning_rate=5)


# In[156]:


clf_ada_sv.fit(X_train, y_train)


# In[157]:


y_pred_ada_sv = clf_ada_sv.predict(X_test)
predictions = [round(value) for value in y_pred_ada_sv]


# In[158]:


# Summary of prediction
print(classification_report(y_test, y_pred_ada_sv))
print(confusion_matrix(y_test, y_pred_ada_sv))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_ada_sv,y_test))


# In[159]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_ada = clf_ada_sv.predict_proba(X_test)
probs_ada = probs_ada[:,1]
auc = roc_auc_score(y_test, probs_ada)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_ada)
plot_roc_curve(fpr, tpr)


# In[189]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_ada = clf_ada_sv.predict_proba(X_test)
probs_ada = probs_ada[:,0]
auc = roc_auc_score(y_test, probs_ada)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_ada)
plot_roc_curve(fpr, tpr)


# # Oversampling AdaBoost Classifier

# In[ ]:


clf_ada_sv_ov = AdaBoostClassifier(n_estimators=50, base_estimator=svc, learning_rate=5)
clf_ada_sv_ov.fit(X_train_over, y_train_over)
y_pred_ada_sv_ov = clf_ada_sv_ov.predict(X_test)
predictions = [round(value) for value in y_pred_ada_sv_ov]

# Summary of prediction
print(classification_report(y_test, y_pred_ada_sv_ov))
print(confusion_matrix(y_test, y_pred_ada_sv_ov))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_ada_sv_ov,y_test))


# In[ ]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve AdaBoost Oversampling class1/Charged off ')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_ada_ov = clf_ada_sv_ov.predict_proba(X_test)
probs_ada_ov = probs_ada_ov[:,0]
auc = roc_auc_score(y_test, probs_ada_ov)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_ada_ov)
plot_roc_curve(fpr, tpr)


# In[ ]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve AdaBoost Oversampling class 0/Fully Paid ')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_ada_ov = clf_ada_sv_ov.predict_proba(X_test)
probs_ada_ov = probs_ada_ov[:,1]
auc = roc_auc_score(y_test, probs_ada_ov)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_ada_ov)
plot_roc_curve(fpr, tpr)


# In[ ]:


Ada Boost Undersampling


# In[ ]:


clf_ada_sv_un = AdaBoostClassifier(n_estimators=50, base_estimator=svc, learning_rate=5)
clf_ada_sv_un.fit(X_train_res, y_train_res)
y_pred_ada_sv_un = clf_ada_sv_un.predict(X_test)
predictions = [round(value) for value in y_pred_ada_sv_un]

# Summary of prediction
print(classification_report(y_test, y_pred_ada_sv_un))
print(confusion_matrix(y_test, y_pred_ada_sv_un))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred_ada_sv_un,y_test))


# In[ ]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.3f'% auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve AdaBoost under sampling class 0/Fully Paid ')
    plt.legend()
    plt.show()

#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
probs_ada_un = clf_ada_sv_un.predict_proba(X_test)
probs_ada_un = probs_ada_un[:,1]
auc = roc_auc_score(y_test, probs_ada_un)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs_ada_un)
plot_roc_curve(fpr, tpr)


# In[ ]:





# # over sampling and under sampling

# In[300]:


# from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

# Over Sampling
ros = RandomOverSampler(random_state=0)
X_train_res, y_train_res = ros.fit_sample(X_train, y_train.ravel())

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[301]:


# Naive Bayes
naive= GaussianNB()
naive.fit(X_train, y_train)

y_pred= naive.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred,y_test))


# In[52]:


# Logistic Regression
log2= LogisticRegression()
log2.fit(X_train, y_train)

y_pred= log2.predict(X_train)

# Summary of the prediction
print(classification_report(y_train, y_pred))
print(confusion_matrix(y_train, y_pred))

# Accuracy
print("Accuracy of the model is: ", accuracy_score(y_pred,y_train))


# In[303]:


tmp = log.fit(X_train_res, y_train_res.ravel())
y_pred_sample_score = tmp.decision_function(X_test)


fpr, tpr, thresholds = roc_curve(y_test, y_pred_sample_score)

roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # smote:undersampling

# In[304]:


# from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

# Under Sampling
rus = RandomUnderSampler(random_state=0)
X_train_res, y_train_res = rus.fit_sample(X_train, y_train.ravel())

print("After UnderSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After UnderSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[305]:


# Naive Bayes
naive= GaussianNB()
naive.fit(X_train, y_train)

y_pred= naive.predict(X_test)

# Summary of prediction
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy score
print("Accuracy of the model is: ", accuracy_score(y_pred,y_test))


# In[307]:


#Logistic Regression
log= LogisticRegression()
log.fit(X_train, y_train)

y_pred= log.predict(X_test)

# Summary of the prediction
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Accuracy
print("Accuracy of the model is: ", accuracy_score(y_pred,y_test))


# In[359]:


tmp = rf.fit(X_train, y_train.ravel())
y_pred_sample_score = tmp.decision_function(X_test)


fpr, tpr, thresholds = roc_curve(y_test, y_pred_sample_score)

roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




