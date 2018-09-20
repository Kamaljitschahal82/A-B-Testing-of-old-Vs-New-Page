
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[77]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[78]:


#read CSV file and store in a df.
df=pd.read_csv('ab_data.csv')
# Read few rows in the dataset:
df.head()


# b. Use the below cell to find the number of rows in the dataset.

# In[79]:


#no of columns & rows are below :
columns = df.shape[1]
print('columns=',columns)
rows = df.shape[0]
print('rows=',rows)


# c. The number of unique users in the dataset.

# In[80]:


# no of unique users 
(df.user_id.nunique())


# d. The proportion of users converted.

# In[81]:


# sum of converted users divided by row counts 

print('The proportion of users converted=',df.converted.sum()/df.shape[0])


# e. The number of times the `new_page` and `treatment` don't line up.

# In[82]:


# rows where group=control & landing page= new_page
df_mismatch_1 = df.query("group == 'control' and landing_page == 'new_page'")

# rows where group= treatment & landing page=  old_page 
df_mismatch_2 = df.query("group == 'treatment' and landing_page == 'old_page'")


# In[83]:


total_mismatch_rows=len(df_mismatch_1)  + len(df_mismatch_2)
print('total_mismatch_rows=',total_mismatch_rows)
# total no of rows for mismatch for both groups are 3893.


# f. Do any of the rows have missing values?

# In[84]:


df.info()
# no missing values as non- null in every column is 294478


# In[85]:


df.isnull().any()
# no missing values 


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[86]:


# drop rows for mismatched for both queries to clean up the data .
df.drop(df.query("group == 'treatment' and landing_page == 'old_page'").index, inplace=True)
df.drop(df.query("group == 'control' and landing_page == 'new_page'").index, inplace=True)
df2=df
df2.info()


# In[87]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[88]:


df2.user_id.nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[89]:


df2[df2.duplicated(['user_id'], keep=False)].user_id


# c. What is the row information for the repeat **user_id**? 

# In[90]:


df2[df2['user_id']==773192]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[91]:


df2.drop(labels=2893, axis=0, inplace=True)


# In[92]:


df2.info()


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[93]:


df2.converted.mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[94]:


df2_grp = df2.groupby('group')
df2_grp.describe()


# # Given that an individual was in the control group, what is the probability they converted? is 0.120386

# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# # Given that an individual was in the treatment group, what is the probability they converted? is 0.118808

# d. What is the probability that an individual received the new page?

# In[95]:


df2['landing_page'].value_counts()[0]/len(df2)


# e. Use the results in the previous two portions of this question to suggest if you think there is evidence that one page leads to more conversions?  Write your response below.

# # Given that an individual was in the control group, what is the probability they converted? is 0.118807
# Given that an individual was in the treatment group, what is the probability they converted? is 0.120386
# 
# According to the probabilities calculated, the control group converted at a higher rate than the teatment version & very small difference between 2 probabilities is 0.2%.
# 
# Many potentially influencing factors are not accounted for : like test durations etc. So, we cannot state that one page leads to more conversions. 
# 
# 

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# # H0 : pold >= pnew -> NULL Hypothesis
#  H1 : pnew > pold -> Alternative hyptothesis
#  

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[96]:


pnew = df2['converted'].mean()
pnew


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[97]:


pold = df2['converted'].mean()
pold


# c. What is $n_{new}$?

# In[98]:


nnew = len(df2.query("group == 'treatment'"))
nnew


# d. What is $n_{old}$?

# In[99]:



nold = len(df2.query("group == 'control'"))
nold


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[100]:


new_page_converted = np.random.choice([1, 0], size=nnew, p=[pnew, (1-pnew)])
len(new_page_converted)


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[101]:


old_page_converted = np.random.choice([1, 0], size=nold, p=[pold, (1-pold)])
len(old_page_converted)


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[102]:


#to calculate p_diff, we can make both coversion having same sizes by truncating the values at 145274.
new_page_converted = new_page_converted[:145274]


# In[103]:


p_diff = (new_page_converted/nnew) - (old_page_converted/nold)
p_diff


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in **p_diffs**.

# In[104]:


p_diffs = []

for _ in range(10000):
    new_page_converted = np.random.choice([1, 0], size=nnew, p=[pnew, (1-pnew)]).mean()
    old_page_converted = np.random.choice([1, 0], size=nold, p=[pold, (1-pold)]).mean()
    diffs = new_page_converted - old_page_converted 
    p_diffs.append(diffs)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[105]:


plt.hist(p_diffs, bins=25)
plt.xlabel('p_diffs')
plt.ylabel('Frequency')
plt.title('10K simulated p_diffs under NULL')
plt.legend()
plt.show();


# # the mean of this normal distribution is o which data indicated as per NULL hypothesis.

# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[106]:


actual_diffs = df[df['group'] == 'treatment']['converted'].mean() -  df[df['group'] == 'control']['converted'].mean()
actual_diffs


# In[107]:


p_diffs = np.array(p_diffs)


# In[108]:


((actual_diffs < p_diffs).mean())*100


# k. In words, explain what you just computed in part **j.**.  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# # Old performed slightly better than new pages . we conclude null hypothesis is true.
# 90.43% indicates newer page will do worse than older page.there is no conversion advantage as indicated by numbers. we calculated that almost 90% of the population in our simulated sample lies above the real difference indicates newer page will do worse than older page.
# we computed p-values here.
# The p-value is the probability of the test statistic being at least as extreme as the one observed given that the null hypothesis is true. A small p-value is an indication that the null hypothesis is false.
# The more extreme in favor of the alternative portion of this statement determines the shading associated with your p-value.
# 
#   
# 

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[109]:


import statsmodels.api as sm
convert_old = sum(df2.query("group == 'control'")['converted'])
convert_new = sum(df2.query("group == 'treatment'")['converted'])
nold = len(df2.query("group == 'control'"))
nnew = len(df2.query("group == 'treatment'"))



# In[110]:


nnew,nold


# In[111]:


df2.head()


# In[112]:


convert_old, convert_new


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[113]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [nold, nnew], alternative='smaller')
z_score, p_value


# In[114]:


z_score1, p_value1 = sm.stats.proportions_ztest([convert_old, convert_new], [nold, nnew], alternative='two-sided')
z_score1, p_value1


# In[115]:


from scipy.stats import norm


# In[116]:


norm.cdf(z_score)


# In[117]:


norm.ppf(1-(0.05/2))
# with a two-tail test -this Tells us what our critical value at 95% confidence is


# In[118]:


norm.ppf(1-(0.05))
# with a two-tail test -this Tells us what our critical value at 95% confidence is


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# # Since the z-score of 1.3109241984234394 doesnt  exceeds the critical value of 1.959( two sided) or 1.644 (one sided), we accept the null hypothesis .so old page is slightlty better than new page
# 
# yes we agree with findings in j and k .
# 
#  

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# Logistics

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a colun for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[119]:


df2['intercept']=1
df2['ab_page'] = pd.Series(np.zeros(len(df2)), index=df2.index)


# In[120]:


index_change_for_treatment = df2[df2['group']=='treatment'].index


# In[121]:


df2.set_value(index=index_change_for_treatment, col='ab_page', value=1)


# In[122]:


df2[[ 'ab_page']] = df2[[ 'ab_page']].astype(int)
df2[df2['group']=='treatment'].head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[123]:



import statsmodels.api as sm
logit = sm.Logit(df2['converted'],df2[['intercept','ab_page']])
results = logit.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[124]:


results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in the **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# # p-value associated with ab_page(treatment) is 0.19, which is slightly lower than the p-value calculated using the z-test section II (a one-tailed test (p_new >p_old)).as added intercept account for error and it is close to true p-value.
# p-value is too high to reject the null hypothesis.
# 
# What are the null and alternative hypotheses associated with your regression model:
# H0: pnew - pold = 0 &&
# 
# H1: pnew - pold != 0

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# # any disadvantages to adding additional terms into your regression model? yes when do you regression analysis we would like to consider large impacts features , on the other side small impacts are usually not influencial and should be left for the intercept.
# we should acount for factors like new skills/ Timestamps with region/weather patterns/ seasons/ age / course type/ previous grades which can alter the overall results.
# 

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy varaibles.** Provide the statistical output as well as a written response to answer this question.

# In[125]:


count_df = pd.read_csv('./countries.csv')


# In[126]:


count_df.head()


# In[127]:


df_dummy = pd.get_dummies(data=count_df, columns=['country'])
df_dummy.head()


# In[128]:


df3 = df_dummy.merge(df2, on='user_id')


# In[131]:



df3.head()


# In[132]:


logit_countries = sm.Logit(df3['converted'], 
                           df3[['country_UK', 'country_US', 'intercept']])


# In[133]:


result2 = logit_countries.fit()


# In[139]:


result2.summary2()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[140]:


logit_countries2 = sm.Logit(df3['converted'], 
                           df3[['country_UK', 'country_US', 'ab_page','intercept']])


# In[141]:


result3 = logit_countries2.fit()


# In[147]:


result3.summary2()


# In[148]:


np.exp(result2.params)


# In[149]:


1/_


# In[150]:


df3.groupby('group').mean()['converted']


# # conclusion;
#     The performance of old page is better after evaulating results . 
#     Null Hypothesis is accepted  and Reject the Alternate Hypothesis.Hence , keep using existing page as it is.
#     even country will impact the conversion rate but it is not statiscally signficant .
#     Histrogram shows worse results for new page compare to exisiting page.
#     
# 

# <a id='conclusions'></a>
# ## Finishing Up
# 
# > Congratulations!  You have reached the end of the A/B Test Results project!  This is the final project in Term 1.  You should be very proud of all you have accomplished!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[151]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

