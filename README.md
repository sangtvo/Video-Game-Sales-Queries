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

* In the First Normal Form (1NF), there are no groups that repeats the data. 
* In the Second Normal Form (2NF), 1NF applies and non-key attributes are fully dependent on its primary key.
* In the Third normal Form (3NF), 1NF and 2NF applies as well as all of its attributes are directly dependent on the primary key. 

In order to quickly separate the data into 3 tables, Microsoft Excel was used. Then Python was used to generate a unique ID for each table (titles, sales, scores) using UUID library.
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

Once each table has a foreign key and primary key, we can proceed with database implementation.

The database schema is as follows:
![db_schema](https://github.com/sangtvo/Video-Game-Sales-Queries/blob/main/images/vg_db_schema.PNG?raw=true)



Database Implementation
---
The example code for DB creation:
```python
import sqlite3
import pandas as pd
sqlite3.connect('vg_sales.db')

conn = sqlite3.connect('vg_sales.db') 
c = conn.cursor() 

# Create table - TITLES
c.execute('''CREATE TABLE TITLES
             ([unique_id] TEXT PRIMARY KEY, [Name] text, [Platform] text, [Year_of_Release] integer, [Genre] text, [Publisher] text, [Developer] text, [Rating] text)''')
          
# Create table - SALES
c.execute('''CREATE TABLE SALES
             ([sales_id] text PRIMARY KEY, [unique_id] text, [NA_Sales] integer, [EU_Sales] integer, [JP_Sales] integer, [Other_Sales] integer, [Global_Sales] integer)''')
        
# Create table - SCORES
c.execute('''CREATE TABLE SCORES
             ([scores_id] text, [unique_id] text, [Critic_Score] integer, [Critic_Count
] integer, [User_Score] float, [User_Count] integer)''')
                 
conn.commit()

# Insert the values from the csv file into the table 'TITLES' 
read_titles = pd.read_csv ('vg_titles.csv')
read_titles.to_sql('TITLES', conn, if_exists='append', index = False) 

read_sales = pd.read_csv ('vg_salesID.csv')
read_sales.to_sql('SALES', conn, if_exists='replace', index = False) 

read_scores = pd.read_csv ('vg_scoresID.csv')
read_scores.to_sql('SCORES', conn, if_exists='replace', index = False) 
```

In order to check if the database was created successfully, a function is created to show the tables in the vg_sales.db.
```python
db = 'vg_sales.db'

# Function that calls the run_query() function to return a list of all tables and views in the database. 
def show_tables():
    q = '''
    SELECT
        name,
        type
    FROM sqlite_master
    WHERE type IN ("table","view");
    '''
    return run_query(q)
# Run the function below to see all the tables and views in vg_sales.db
show_tables()
```
```
	name	type
0	TITLES	table
1	SALES	table
2	SCORES	table
```

After checking the columns for each table with SELECT *, we will create two more functions.
```python
# Function that creates a SQL query and returns a pandas dataframe.
def run_query(q):
    with sqlite3.connect(db) as conn:
        return pd.read_sql(q, conn)
    
# Function that takes a SQL command as argument and executives it using the sqlite module.
def run_command(c):
    with sqlite.connect(db) as conn:
        conn.isolation_level = None 
        conn.execute(c)
```

Now, we can query our own database and start answering several business questions.

SQL Queries
---
What is the percentage of all sales for each genre?
```sql
q1 = '''
SELECT t.genre, COUNT(*) sales, COUNT(*) * 100.0 / (SELECT COUNT(*) FROM sales) sales_percentage
FROM titles t
INNER JOIN sales s
    ON s.unique_id = t.unique_id
GROUP BY 1
ORDER BY 2 DESC
'''
genre_sales = run_query(q1)
genre_sales
```
```
	Genre	sales	sales_percentage
0	Action	3370	20.156708
1	Sports	2348	14.043902
2	Misc	1750	10.467133
3	Role-Playing	1500	8.971828
4	Shooter	1323	7.913153
5	Adventure	1303	7.793528
6	Racing	1249	7.470542
7	Platform	888	5.311322
8	Simulation	874	5.227585
9	Fighting	849	5.078055
10	Strategy	683	4.085173
11	Puzzle	580	3.469107
12	None	2	0.011962
```
```python
ax = sns.barplot(x='sales', y='Genre', data=genre_sales, palette='cool')
ax.set_title('Top Selling Genres', size=14, weight='bold')
ax.set_xlabel('Total Sales')
ax.set_ylabel('Genre')
plt.show()
```

![top_genres](https://github.com/sangtvo/Video-Game-Sales-Queries/blob/main/images/top_genres.png?raw=true)

* The top selling genre worldwide are action titles which makes up 20.15% of the sales, followed by sports and miscelanneous titles. We can see that most consumers prefer not to play any mind boggling genres which is roughly 4-5%. 


Key Takeaways
---
* Based on the data, the churn rate of the data is 26.58% and more than half of the data are month-to-month contract customers. 
  * In turn, month to month churn rate is extremely high compared to 1-2 year contracts.
* ~$20-25 monthly rate is an extremely common charge for most customers.
* Tenure, contract, and internet service are important predictors for churn under step-wise logistic regression and decision tree models.
* Total charges is an important variable for random forest model, but not step-wise logistic regression and decision tree models.
  * Top 3 variables that are important for churn rate are tenure, contract, and total charges.
* Random forest model outperforms the other models with an overall accuracy of 80.21%.
