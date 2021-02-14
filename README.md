# Project 06: Video Games Sales Queries
> This project analyzes video game sales data from [Kaggle](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings) by answering several possible business questions using SQL. From start to finish, this project normalizes the data to create a database, queries the database, and visualize the data. 

Table of Contents
---
1. [General Information](#general-information)
2. [Summary](#summary)
3. [Tech Stack](#tech-stack)
4. [Data Preprocessing/Cleaning](#data-preprocessingcleaning)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Logistic Regression (Base)](#logistic-regression)
7. [Step-wise Logistic Regression](#step-wise-logistic-regression)
8. [Decision Tree](#decision-tree)
9. [Random Forest](#random-forest)
10. [Solution](#solution)
11. [Key Takeaways](#key-takeaways)

<a name="https://github.com/sangtvo/Customer-Churn-Analysis#general-information"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#summary"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#tech-stack"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#data-preprocessingcleaning"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#exploratory-data-analysis"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#logistic-regression"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#step-wise-logistic-regression"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#decision-tree"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#random-forest"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#solution"/>
<a name="https://github.com/sangtvo/Customer-Churn-Analysis#key-takeaways"/>

General Information
---

Summary
---
The winning model is the **random forest** algorithm with an overall accuracy of 80.21% and AUC of 81.50%. This means that the model will correctly predict churn 80.21% of the time and there is a 81.50% chance that the model can distinguish between positive (churn) and negative (no churn) classes. In order for the telecommunications company to reduce the current churn rate of 26.58%, the company should focus on contracts (month-to-month specifically), tenure length, and total charges to start. The company should give incentives such as reduced pricing or discounted extra services for long-term customers to keep them from leaving to another competitor. 

Tech Stack
---
* Python
    * NumPy
    * Pandas
    * Seaborn
    * UUID
* Microsoft Excel
* VS Code
* SQLite3

Data Normalization
---
The data consists of 16 attributes and 16,720 variables. Before a database is created, data normalization needs to be implemented. The advantages of normalizing data is that it will create consistency and it will be much easier to map objects from one table to the next.

For this particular database, we will normalize up to the third normal form (3NF).

In the First Normal Form (1NF), there are no groups that repeats the data. 
In the Second Normal Form (2NF), 1NF applies and non-key attributes are fully dependent on its primary key.
In the Third normal Form (3NF), 1NF and 2NF applies as well as all of its attributes are directly dependent on the primary key. 

In order to quickly separate the data into 3 tables, Microsoft Excel was used. Then Python was used to generate a unique ID for each table (titles, sales, scores). 
```python
import pandas as pd
import uuid
df = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")

# Creates unique ID for each row.
df['unique_ID'] = [uuid.uuid4().hex for i in range(len(df.index))]

# Checking if the new column created unique IDs.
df['unique_ID'].unique()

# Export sales table into CSV
df.to_csv('vg_salesID.csv', index=False)
```

Once each table has a foreign key and primary key, it is ready to implement the database.

The database schema is as follows:


Exploratory Data Analysis
---

<table>
  <tr><th colspan=2>Univariate Analysis</th></tr>
  <tr><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/bar_1.png?raw=true"> </td><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/bar_2.png?raw=true"> </td></tr>
  <tr><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/bar_3.png?raw=true"> </td><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/bar_4.png?raw=true"> </td></tr>
</table>

* ***Gender*** and ***Partner*** are almost evenly split 50/50 and that 83.76% of the data are not ***SeniorCitizens***. 
* 90.33% of the data have single ***PhoneService*** lines, but in terms of extra services, there are roughly 70% of the customers that do not have ***OnlineSecurity*** and ***TechSupport***. 
* While ***InternetService*** is a pretty common add-on with phone lines, **fiber optic** is a favorable internet service which accounts for 44.03% of the customers. 
* 55.11% of the customers are on a **month-to-month** ***Contract*** and 33.63% of the customers pay their bill with an **electronic check**.
* The ***Churn*** rate of the data is 26.58%.


<table>
  <tr><th colspan=2>Bivariate Analysis</th></tr>
  <tr><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/bivar_1.png?raw=true"> </td><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/bivar_2.png?raw=true"> </td></tr>
  <tr><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/bivar_3.png?raw=true"> </td><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/bivar_4.png?raw=true"> </td></tr>
  <tr><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/bivar_5.png?raw=true"> </td>
</table>

* ***Gender*** percentages are almost similar and therefore, may not have influence on churn. 
* ***OnlineSecurity*** and ***TechSupport*** have very similar percentages and perhaps they are somewhat correlated whether a customer is churning or not.
* ***OnlineBackup*** and ***DeviceProtection*** are also similar indicating that they might be correlated with one another.
* Customers with ***PhoneService*** are more likely to churn those who don't have service.
* **Month-to-month** ***Contract*** customers are much higher than longer contracts, which makes sense because there is no obligation to stay longer if it was a month-to-month basis and can leave at any time. 
* Customers with ***InternetService*** are more likely to churn than those that don't have internet.
  * **Fiber optic** churn rate is 3x higher than **DSL** and 9x without internet service.
* Churn percentage is higher for customers who utilizes ***PaperlessBilling*** option.
* Customers who use **electronic checks** are almost 5x as high to churn compared to other ***PaymentMethod*** options.

<table>
  <tr><th colspan=2>Distribution of Continuous Independent Variables</th></tr>
  <tr><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/distribution_MonthlyCharges.png?raw=true"> </td><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/distribution_TotalCharges.png?raw=true"> </td></tr>
  <tr><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/distribution_tenure.png?raw=true"> </td><td>
</table>

* The number of customers who have ***MonthlyCharges*** of $25 or less are extremely high--it seems that $0-25 is a very common monthly charge.  
  * The distributions starting around $30 per month is similar between customers who churned or not.
* ***TotalCharges*** has a positive/right skew meaning that the right side of the distribution is longer or flatter. 
  * Mean and median is greater than the mode.
  * The frequency count at $2,500 total charges starts to decrease slowly in count and not as rapid. 
* ***Tenure*** distributions are different between customers who churned or not. 
  * For customers who churned, the distribution is positive/right skew which may indicate that the first 10 weeks or so, customers are more likely to cancel service.
  * For customers who did not churn, the highest peak is 70 weeks (almost 6 years) which indicates that there is a large amount of customers who kept their service for at least 5 years. 

![Correlation](https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/correlation.png?raw=true)

* ***TotalCharges*** and ***tenure*** have a positive correlation which means that the correlation coefficient is greater than 0, but not a perfect correlation of 1.0 as colored in the correlation matrix above.
  * Correlation coefficients are indicators of the strength of the linear relationship between two variables. 
  * While there is a positive correlation, it does not mean that one causes the other. 

<table>
  <tr><th colspan=2>Boxplots of Continuous Independent Variables</th></tr>
  <tr><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/bp_MonthlyCharges.png?raw=true"> </td><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/bp_TotalCharges.png?raw=true"> </td></tr>
  <tr><td> <img src="https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/bp_tenure.png?raw=true"> </td><td>
</table>

* There are no obvious outliers in the continuous variables where there are values beyond the whiskers of the boxplots. 

Logistic Regression
---
(Binary) logistic regression is a common classification algorithm when the categorical response have only two possible outcomes, in this case, churn or no churn. This involves regressing the predictor variables on a binary outcome using a binomial link function.

First, the data frame is split 70% training and 30% testing data.
```r
set.seed(123)
train_test_data <- createDataPartition(cdf$Churn,p=0.7,list=FALSE)
train_d <- cdf[train_test_data,]
test_d <- cdf[-train_test_data,]
```

Run the base logistic regression model with all variables.
```r
lr_model <- glm(Churn ~., data=train_d, family=binomial(link='logit'))
summary(lr_model)
```
```
Call:
glm(formula = Churn ~ ., family = binomial(link = "logit"), data = train_d)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.8637  -0.6953  -0.2890   0.7553   3.2200  

Coefficients:
                                       Estimate Std. Error z value Pr(>|z|)    
(Intercept)                           1.0746695  0.9571525   1.123 0.261532    
genderMale                           -0.0482269  0.0771304  -0.625 0.531797    
SeniorCitizenYes                      0.1200467  0.1014491   1.183 0.236683    
PartnerYes                           -0.0910761  0.0918164  -0.992 0.321228    
DependentsYes                        -0.0843260  0.1068161  -0.789 0.429849    
tenure                               -0.0621135  0.0074487  -8.339  < 2e-16 ***
PhoneServiceYes                       0.0233358  0.7635434   0.031 0.975618    
MultipleLinesYes                      0.3794100  0.2078464   1.825 0.067935 .  
InternetServiceFiber optic            1.5613737  0.9372764   1.666 0.095741 .  
InternetServiceNo                    -1.5047493  0.9480893  -1.587 0.112481    
OnlineSecurityYes                    -0.2149292  0.2098780  -1.024 0.305804    
OnlineBackupYes                       0.0008494  0.2053088   0.004 0.996699    
DeviceProtectionYes                   0.1177891  0.2084585   0.565 0.572041    
TechSupportYes                       -0.1280168  0.2128673  -0.601 0.547579    
StreamingTVYes                        0.5187839  0.3819512   1.358 0.174386    
StreamingMoviesYes                    0.5783808  0.3840289   1.506 0.132045    
ContractOne year                     -0.8451411  0.1304035  -6.481 9.11e-11 ***
ContractTwo year                     -1.4644194  0.2039949  -7.179 7.04e-13 ***
PaperlessBillingYes                   0.3020753  0.0884590   3.415 0.000638 ***
PaymentMethodCredit card (automatic) -0.0093796  0.1357499  -0.069 0.944914    
PaymentMethodElectronic check         0.2897469  0.1124649   2.576 0.009985 ** 
PaymentMethodMailed check            -0.1002695  0.1367827  -0.733 0.463524    
MonthlyCharges                       -0.0336622  0.0372618  -0.903 0.366314    
TotalCharges                          0.0003653  0.0000844   4.328 1.50e-05 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 5702.8  on 4923  degrees of freedom
Residual deviance: 4117.1  on 4900  degrees of freedom
AIC: 4165.1

Number of Fisher Scoring iterations: 6
```
* Variables that are significant are: tenure, contract, payment method electronic check, and total charges.

Akaike Information Criterion (AIC) evaluates how well a model fits the data without over-fitting. The AIC score rewards models that have a high goodness-of-fit score, but penalizes when models become more complex. AIC score is useful when comparing AIC scores of competing models and the lower the AIC score, the better the fit (more parsimonious model). 

Compute the confusion matrix table.
```r
lr_prob1 <- predict(lr_model, test_d, type="response")
lr_pred1 <- ifelse(lr_prob1 > 0.5,"Yes","No")

table(Predicted=lr_pred1, Actual=test_d$Churn)
```
```
         Actual
Predicted   No  Yes
      No  1385  229
      Yes  163  331
```
```
         Actual
Predicted   No  Yes
      No    TN  FN
      Yes   FP  TP
```
* TP = true positive; cases we predicted they churned and indeed churned.
* TN = true negative; cases we predicted they did not churn and did not churn.
* FP = false positive (type I error); cases we predicted they did churn, but did not actually churned.
* FN = false negative (type II error); cases we predicted they did not churn, but they actually churned.
```
Specificity = TN/(TN + FP)
Sensitivity = TP / (FN + TP)
Negative Predicted Value (-PV) = TN/(TN+FN)
Positive Predicted Value (+PV or precision) = TP/(FP+TP)
```
* Specificity = 1385/(1385+163) = 0.8947
* Sensitivity = 331/(229+331) = 0.5910
* -PV = 1385/(1385+229) = 0.8581
* +PV = 331/(163+331) = 0.6700

Validation measures are specificity and sensitivity. Sensitivity is the true positive rate where the proportion of test in question is correctly classified. Specificity is the true negative rate where the proportion of negative test in question is correctly classified. Sensitivity and specificity are inversely proportional, meaning that as sensitivity increases, the specificity decreases and vice versa. 

Positive predicted value, or precision, tells how often a positive test represents a true positive. Negative predicted value tells how often a negeative test represents a true negative. 

Calculate the accuracy.
```r
lr_prob2 <- predict(lr_model, train_d, type="response")
lr_pred2 <- ifelse(lr_prob2 > 0.5,"Yes","No")
lr_tab1 <- table(Predicted = lr_pred2, Actual = train_d$Churn)
lr_tab2 <- table(Predicted = lr_pred1, Actual = test_d$Churn)
lr_accuracy <- sum(diag(lr_tab2))/sum(lr_tab2)
lr_accuracy
```
```
[1] 0.8140417
```

Calculcate the Receiver Operating Character (ROC) Curve.
```r
lr_roc <- roc(train_d$Churn, lr_prob2)
plot(lr_roc, col='Blue', main="ROC Curve")
auc(lr_roc)
```
```
Area under the curve: 0.8446
```
![BaseLR](https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/base_lr_roc.png?raw=true)

ROC is a probability curve that shows the performance of a classification model at all classification thresholds. The x-axis is the false positive rate and the y-axis is the true positive rate. The AUC represents the measure of separability that shows how much the model can distinguish between classes. The higher the AUC, the better at predicting 0s and 1s. AUC ranges in value from 0 to 1 and a model  with predictions are 100% wrong has an AUC of 0 and predictions that are 100% correct has an AUC of 1.

The accuracy of the base regression model is 81.40% with AUC of 84.46%. The model has a 84.46% chance that it will distinguish between a positive and negative class. 

Step-wise Logistic Regression
---
We can improve the base model by using the stepAIC for variable selection. This method will add or remove variables and comes up with a final set of variables. This method simplifies the model and gives the most parsimonious model. 

```r
lrstep_model <- MASS::stepAIC(lr_model, trace=0)
summary(lrstep_model)
```
```
Call:
glm(formula = Churn ~ SeniorCitizen + Partner + tenure + MultipleLines + 
    InternetService + OnlineSecurity + StreamingTV + StreamingMovies + 
    Contract + PaperlessBilling + PaymentMethod + MonthlyCharges + 
    TotalCharges, family = binomial(link = "logit"), data = train_d)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.8885  -0.6953  -0.2900   0.7595   3.2334  

Coefficients:
                                       Estimate Std. Error z value Pr(>|z|)    
(Intercept)                           1.027e+00  3.249e-01   3.161 0.001571 ** 
SeniorCitizenYes                      1.436e-01  9.914e-02   1.449 0.147360    
PartnerYes                           -1.189e-01  8.345e-02  -1.425 0.154237    
tenure                               -6.222e-02  7.440e-03  -8.364  < 2e-16 ***
MultipleLinesYes                      3.800e-01  1.022e-01   3.719 0.000200 ***
InternetServiceFiber optic            1.571e+00  2.119e-01   7.411 1.25e-13 ***
InternetServiceNo                    -1.480e+00  2.095e-01  -7.064 1.62e-12 ***
OnlineSecurityYes                    -2.223e-01  1.059e-01  -2.098 0.035877 *  
StreamingTVYes                        5.151e-01  1.145e-01   4.500 6.81e-06 ***
StreamingMoviesYes                    5.792e-01  1.142e-01   5.071 3.97e-07 ***
ContractOne year                     -8.506e-01  1.294e-01  -6.572 4.95e-11 ***
ContractTwo year                     -1.495e+00  2.011e-01  -7.435 1.05e-13 ***
PaperlessBillingYes                   3.041e-01  8.822e-02   3.447 0.000567 ***
PaymentMethodCredit card (automatic) -8.973e-03  1.355e-01  -0.066 0.947215    
PaymentMethodElectronic check         2.931e-01  1.123e-01   2.611 0.009029 ** 
PaymentMethodMailed check            -1.090e-01  1.365e-01  -0.798 0.424728    
MonthlyCharges                       -3.322e-02  6.473e-03  -5.132 2.87e-07 ***
TotalCharges                          3.694e-04  8.407e-05   4.394 1.11e-05 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 5702.8  on 4923  degrees of freedom
Residual deviance: 4121.1  on 4906  degrees of freedom
AIC: 4157.1

Number of Fisher Scoring iterations: 6
```

Checking variance inflation factor (VIF) for **multicollinearity**. Multicollinearity is when an independent variable is highly correlated with one or more of the other independent variables. This is an issue because it undermines the significance of the independent variable. 
```r
rms::vif(lrstep_model)
```
```
                    SeniorCitizenYes                           PartnerYes                               tenure                     MultipleLinesYes 
                            1.101765                             1.126056                            16.066630                             1.740315 
          InternetServiceFiber optic                    InternetServiceNo                    OnlineSecurityYes                       StreamingTVYes 
                            7.288274                             2.719026                             1.282502                             2.163958 
                  StreamingMoviesYes                     ContractOne year                     ContractTwo year                  PaperlessBillingYes 
                            2.151208                             1.289170                             1.317572                             1.141450 
PaymentMethodCredit card (automatic)        PaymentMethodElectronic check            PaymentMethodMailed check                       MonthlyCharges 
                            1.623259                             2.127947                             2.067715                            20.886103 
                        TotalCharges 
                           20.728399 
```

Remove VIF > 10 and re-run the step-wise logistic regression model. I also manually removed the remaining least significant variables as another step-wise process (see R markdown/html for full code):
```
Call:
glm(formula = Churn ~ tenure + InternetService + OnlineSecurity + 
    StreamingTV + StreamingMovies + Contract + PaperlessBilling, 
    family = binomial(link = "logit"), data = train_d)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.7869  -0.6783  -0.3040   0.7767   3.1201  

Coefficients:
                           Estimate Std. Error z value Pr(>|z|)    
(Intercept)                -0.52568    0.09620  -5.464 4.65e-08 ***
tenure                     -0.03262    0.00244 -13.367  < 2e-16 ***
InternetServiceFiber optic  0.92930    0.09092  10.221  < 2e-16 ***
InternetServiceNo          -0.83840    0.14600  -5.742 9.34e-09 ***
OnlineSecurityYes          -0.40620    0.09849  -4.125 3.71e-05 ***
StreamingTVYes              0.27410    0.09214   2.975  0.00293 ** 
StreamingMoviesYes          0.37260    0.09227   4.038 5.39e-05 ***
ContractOne year           -0.97795    0.12628  -7.744 9.63e-15 ***
ContractTwo year           -1.64444    0.19597  -8.391  < 2e-16 ***
PaperlessBillingYes         0.35244    0.08681   4.060 4.91e-05 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 5702.8  on 4923  degrees of freedom
Residual deviance: 4187.3  on 4914  degrees of freedom
AIC: 4207.3

Number of Fisher Scoring iterations: 6
```
* We can see that the most significant variables are tenure, internet service, online security, streaming tv and movies, contract, and paperless billing. 

Confusion matrix for the step-wise logistic regression model:
```
        Actual
Predicted   No  Yes
      No  1386  256
      Yes  162  304
```
* Specificity = 1386/(1386+162) = 0.8953
* Sensitivity = 304/(256+304) = 0.5428
* -PV = 1386/(1386+256) = 0.8440
* +PV = 304/(162+304) = 0.6423

The accuracy of the model is 80.17% and AUC of 83.77%. 

**Comparison Between Base Logistic Regression & Step-Wise Logistic Regression**
Logistic Regression (Base) ROC Curve             |  Step-wise Logistic Regression ROC Curve
:-------------------------:|:-------------------------:
![BaseLR](https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/base_lr_roc.png?raw=true)  |  ![StepLR](https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/step_roc.png?raw=true)
AUC = 84.46% | AUC = 83.77%
Accuracy = 81.40% | Accuracy = 80.17%

Between the two models, the most parsimonious model would be the step-wise model despite a lower accuracy than the base model. This model is preferrable since it only includes the least variables that are highly significant with a lower AIC score. 

Decision Tree
---
Decision tree is a supervised machine learning algorithm that uses a tree-like model of decisions. 

Run the decision tree.
```r
dtree_model <- rpart(Churn ~., data=train_d, method="class")
rpart.plot(dtree_model)
```
![Dtree](https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/dtree.png?raw=true)

* The root node, contract, shows that contracts that month-to-month contracts are more likely to churn compared to contracts that are 1-2 years.
* Customers with DSL as internet service service are less likely to churn.
* Customers that have no internet service are less likely to churn.
* Customers that stayed with the company for less than 14 months are more likely to churn.

```r
dtree_prob1 <- predict(dtree_model, test_d)
dtree_pred1 <- ifelse(dtree_prob1[,2] > 0.5, "Yes", "No")
table(Predicted=dtree_pred1, Actual=test_d$Churn)
```
```
         Actual
Predicted   No  Yes
      No  1384  279
      Yes  164  281
```
* Specificity = 1384/(1384+164) = 0.8940
* Sensitivity = 281/(279+281) = 0.5017
* -PV = 1384/(1384+279) = 0.8322
* +PV = 281/(164+281) = 0.6314

Calculate the accuracy of the decision tree.
```r
dtree_prob2 <- predict(dtree_model, train_d)
dtree_pred2 <- ifelse(dtree_prob2[,2] > 0.5,"Yes","No")
dtree_tab1 <- table(Predicted=dtree_pred2, Actual=train_d$Churn)
dtree_tab2 <- table(Predicted=dtree_pred1, Actual=test_d$Churn)
dtree_accuracy <- sum(diag(dtree_tab2))/sum(dtree_tab2)
dtree_accuracy
```
```
[1] 0.7898482
```

Create the ROC curve and calculcate the AUC.
```r
dtree_roc <- roc(train_d$Churn, dtree_prob2[,2])
plot(dtree_roc, col='Blue', main="ROC Curve")
auc(dtree_roc)
```
```
Area under the curve: 0.8125
```
![DtreeROC](https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/dtree_roc.png?raw=true)

The decision tree model's accuracy of 78.98% has decreased compared to the step-wise logistic regression model, 80.17%. This model can correctly predict the churn rate for customers 78.98% of the time. 

Random Forest
---
Random forest is a supervised machine learning algorithm that builds a large number of decision trees and merges together to get a more accurate prediction.

Run the random forest. 
```r
rf_model <- randomForest(Churn ~., data=train_d, ntree=50, mtry=2, method="rf", importance=TRUE)
rf_model
```
* As I adjusted the `mtry` parameter (number of variables randomly sampled as candidates at each split) and noticed the AUC was increasingly high in the ROC curve. The value 2 seemed much more ideal for `mtry`.
```
Call:
 randomForest(formula = Churn ~ ., data = train_d, ntree = 50,      mtry = 2, method = "rf", importance = TRUE) 
               Type of random forest: classification
                     Number of trees: 50
No. of variables tried at each split: 2

        OOB estimate of  error rate: 20.13%
Confusion matrix:
      No Yes class.error
No  3306 309  0.08547718
Yes  682 627  0.52100840
```

Run the variable importance plot to show the important drivers of churn.
```r
varImpPlot(rf_model, sort=T, n.var=10, main='Top 10 Variables')
```
![Top10](https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/Top10_Variables.png?raw=true)

The first plot, mean decrease accuracy, is based on how much the accuracy decreases if the variable is excluded. 
The second plot, mean decrease Gini, is based on the decrease of Gini impurity (probability of misclassifying an observation) if a variable is chosen to split a node. 

*  Top 3 important predictors for churn are: tenure, contract, and total charges.

Calculcate the confusion matrix.
```r
rf_pred1 <- predict(rf_model, test_d)
table(Predicted=rf_pred1, Actual=test_d$Churn)
```
```
         Actual
Predicted   No  Yes
      No  1416  285
      Yes  132  275
```
* Specificity = 1416/(1416+132) = 0.9147
* Sensitivity = 275/(275+285) = 0.4910
* -PV = 1416/(1416+2285) = 0.8324
* +PV = 275/(132+275) = 0.6756

Calculcate accuracy.
```r
rf_prob2 <- predict(rf_model, train_d)
rf_pred2 <- ifelse(as.numeric(rf_prob2) > 0.5,"Yes","No")
rf_tab1 <- table(Predicted=rf_pred2, Actual=train_d$Churn)
rf_tab2 <- table(Predicted=rf_pred1, Actual=test_d$Churn)
rf_accuracy <- sum(diag(rf_tab2))/sum(rf_tab2)
rf_accuracy
```
```
[1] 0.8021822
```

Create the ROC curve and calculcate the AUC.
```r
rf_roc <- roc(train_d$Churn, predictor=as.numeric(rf_prob2))
plot(rf_roc, col='Blue', main="ROC Curve")
auc(rf_roc)
```
```
Area under the curve: 0.815
```
![rfROC](https://github.com/sangtvo/Customer-Churn-Analysis/blob/main/images/rf_roc.png?raw=true)

The overall accuracy of the random forest model is 80.21% which performs better than the decision tree model and the step-wise logistic regression model.

Solution
---
MODELS | (Base) Logistic Regression |  Step-wise Logistic Regression | Decision Tree | Random Forest 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
Specificity | 0.8947 | 0.8953 | 0.8940 | 0.9147
Sensitivity | 0.5910 | 0.5428 | 0.5017 | 0.4910
-PV | 0.8581 | 0.8440 | 0.8322 | 0.8324
+PV | 0.6700 | 0.6423 | 0.6314 | 0.6756
AUC | 0.8446 | 0.8377 | 0.8125 | 0.8150
Accuracy | 81.40% | 80.17% | 78.98% | 80.21%

* The best model is the **random forest model** with an overall accuracy of 80.21% with an area under the curve of 81.50% meaning the model will correctly predict churn 80.21% of the time and there is a 81.50% chance that the model can distinguish between positive (churn) and negative (no churn) classes. 

In a business perspective, it is more fitting to tackle the most important variables that causes customers to leave. If the telecommunications wanted to reduce their churn rate, they would need to look into **tenure** length, **contract** length (month-to-month specifically), and **total charges** to start off based on the top 3 variables of the random forest model. Perhaps the company needs to lower the pricing or give incentives for customers to stay longer such as discounted extra services. While there may be some other indicators, it is a good objective as a business to focus the larger and easier predictors to mitigate the loss.   

Key Takeaways
---
* Based on the data, the churn rate of the data is 26.58% and more than half of the data are month-to-month contract customers. 
  * In turn, month to month churn rate is extremely high compared to 1-2 year contracts.
* ~$20-25 monthly rate is an extremely common charge for most customers.
* Tenure, contract, and internet service are important predictors for churn under step-wise logistic regression and decision tree models.
* Total charges is an important variable for random forest model, but not step-wise logistic regression and decision tree models.
  * Top 3 variables that are important for churn rate are tenure, contract, and total charges.
* Random forest model outperforms the other models with an overall accuracy of 80.21%.
