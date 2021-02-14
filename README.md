# Project 06: Video Games Sales Queries
> This project analyzes video game sales data from [Kaggle](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings) by answering several possible business questions using SQL. From start to finish, this project normalizes the data to create a database, queries the database, and visualize the data. 

Table of Contents
---
1. [General Information](#general-information)
2. [Summary](#summary)
3. [Tech Stack](#tech-stack)
4. [Data Normalization](#data-normalization)
5. [Database Implementation](#database-implementation)
6. [SQL Queries](#sql-queries)
    * [What is the percentage of all sales for each genre?](#what-is-the-percentage-of-all-sales-for-each-genre)
7. [Step-wise Logistic Regression](#step-wise-logistic-regression)
8. [Decision Tree](#decision-tree)
9. [Random Forest](#random-forest)
10. [Solution](#solution)
11. [Key Takeaways](#key-takeaways)

<a name="https://github.com/sangtvo/Video-Game-Sales-Queries#general-information"/>
<a name="https://github.com/sangtvo/Video-Game-Sales-Queries#summary"/>
<a name="https://github.com/sangtvo/Video-Game-Sales-Queries#tech-stack"/>
<a name="https://github.com/sangtvo/Video-Game-Sales-Queries#data-normalization"/>
<a name="https://github.com/sangtvo/Video-Game-Sales-Queries#database-implementation"/>
<a name="https://github.com/sangtvo/Video-Game-Sales-Queries#sql-queries"/>
<a name="https://github.com/sangtvo/Video-Game-Sales-Queries#what-is-the-percentage-of-all-sales-for-each-genre"/>
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
Due to SQL query output not aligning correctly on github, output will be displayed in the code folder.

### What is the percentage of all sales for each genre?
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
```python
ax = sns.barplot(x='sales', y='Genre', data=genre_sales, palette='cool')
ax.set_title('Top Selling Genres', size=14, weight='bold')
ax.set_xlabel('Total Sales')
ax.set_ylabel('Genre')
plt.show()
```

![top_genres](https://github.com/sangtvo/Video-Game-Sales-Queries/blob/main/images/top_genres.png?raw=true)

* The top selling genre worldwide are action titles which makes up 20.15% of the sales, followed by sports and miscelanneous titles. We can see that most consumers prefer not to play any mind boggling genres which is roughly 4-5%. 

### Who are the top 10 publishers?
```sql
q2 = '''
SELECT publisher, COUNT(name) num_of_games, COUNT(*) * 100.0 / (SELECT COUNT(*) FROM titles) total_game_percentage
FROM titles
GROUP BY publisher
ORDER BY num_of_games DESC
LIMIT 10
'''
top_publishers = run_query(q2)
top_publishers
```
```python
ax2 = sns.barplot(x='num_of_games', y='Publisher', data=top_publishers, palette='plasma')
ax2.set_title('Top 10 Publishers', size=14, weight='bold')
ax2.set_xlabel('Total Games')
ax2.set_ylabel('Publisher')
plt.show()
```

![top_pub](https://github.com/sangtvo/Video-Game-Sales-Queries/blob/main/images/top_publishers.png?raw=true)

* Electronic Arts takes the lead for the most titles be created and sold globally which accounts for 8.11% of the data. All publishers starting THQ do not make titles half as much as Electronic Arts does.

### What are the sales per region?
```sql
q3 = '''
SELECT SUM(s.na_sales) na_sales, SUM(s.eu_sales) eu_sales, SUM(s.jp_sales) jp_sales, SUM(s.other_sales) other_sales, SUM(s.global_sales) global_sales
FROM sales s
INNER JOIN titles t
    ON t.unique_id = s.unique_id
'''
region_sales = run_query(q3)
region_sales
```
```
na_sales	eu_sales	jp_sales	other_sales	global_sales
4402.62	        2424.67	    1297.43	    791.34	    8920.3
```
* It seems that North America consumes half of the game sales and that a gaming lifestyle is quite prominent in this region. 

### What is the average critic and user score for the top 5 publishers?
```sql
q4 = '''
SELECT t.publisher, AVG(s.critic_score)/10 avg_critic_score, AVG(s.user_score) avg_user_score
FROM scores s
INNER JOIN titles t
    ON s.unique_id = t.unique_id
WHERE t.publisher IN ('Electronic Arts', 'Activision', 'Namco Bandai Games', 'Ubisoft', 'Konami Digital Entertainment')
GROUP BY t.publisher
'''
avg_scores = run_query(q4)
avg_scores
```
```
	Publisher	                    avg_critic_score	avg_user_score
0	Activision	                    6.966784	        4.927842
1	Electronic Arts	                    7.447619	        6.330500
2	Konami Digital Entertainment	    6.834451	        5.188391
3	Namco Bandai Games	            6.644803	        6.155769
4	Ubisoft	                        6.851434	            4.974619
```
```python
# To melt the two columns into observation data
m_avg_score = pd.melt(avg_scores, id_vars = 'Publisher')

ax3 = sns.catplot(x='Publisher', y='value', hue='variable', data=m_avg_score, kind='bar', aspect=2, palette='Blues')
plt.show()
```
![avg_score](https://github.com/sangtvo/Video-Game-Sales-Queries/blob/main/images/avg_scores.png?raw=true)
* Critic scores tend to be much higher on average compared to user scores. Perhaps gamers expect high quality content and have much higher standards compared to critics.

### Which year has the most releases?
```sql
q5 = '''
SELECT year_of_release, COUNT(*) num_of_titles
FROM titles
GROUP BY year_of_release
ORDER BY year_of_release ASC
'''
release_yr_by_title = run_query(q5)
release_yr_by_title
```
```python
# Convert years as categorical in order for data to be visualized.
release_yr_by_title['Year_of_Release'] = release_yr_by_title['Year_of_Release'].astype('category')

ax4 = sns.lineplot(x='Year_of_Release', y='num_of_titles', data=release_yr_by_title)
ax4.set_title('Releases Over the Years', size=14, weight='bold')
ax4.set_xlabel('Years')
ax4.set_ylabel('Number of Titles')
plt.show()
```
![releases_over_yrs](https://github.com/sangtvo/Video-Game-Sales-Queries/blob/main/images/releases_over_yrs.png?raw=true)

* The peak number of title releases was in 2008 that had 1,427 titles. Roughly in the years 2008-2011, publishers were in full force and created many titles for consumers. However, it declined over the next few years.

### What year had the most North America Sales?
```sql
q6 = '''
SELECT year_of_release, SUM(na_sales) NA_sales
FROM titles t
INNER JOIN sales s
    ON s.unique_id = t.unique_id
GROUP BY year_of_release
ORDER BY NA_sales DESC
'''
release_yr_by_sales = run_query(q6)
release_yr_by_sales
```
```python
release_yr_by_sales['Year_of_Release'] = release_yr_by_sales['Year_of_Release'].astype('category')
ax5 = sns.lineplot(x='Year_of_Release', y='NA_sales', data=release_yr_by_sales)
ax5.set_title('Yearly NA Sales', size=14, weight='bold')
ax5.set_xlabel('Years')
ax5.set_ylabel('Sales')
plt.show()
```
![na_sales](https://github.com/sangtvo/Video-Game-Sales-Queries/blob/main/images/yearly_na_sales.png?raw=true)
* As expected, 2008 was the year publishers created the most titles and therefore, had the most sales. The distribution of NA sales have a similar trend with the number of titles over the years. 

### Which platform is most popular?
```sql
q7 = '''
SELECT t.platform, SUM(global_sales) global_sales
FROM titles t
INNER JOIN sales s
    ON s.unique_id = t.unique_id
GROUP BY platform
ORDER BY global_sales DESC
'''
platform_sales = run_query(q7)
platform_sales
```
```python
plt.figure(figsize=(14,5))
ax6 = sns.barplot(x='Platform', y='global_sales', data=platform_sales, palette='RdPu_r')
ax6.set_title('Platform Global Sales', size=14, weight='bold')
ax6.set_xlabel('Platform')
ax6.set_ylabel('Sales')
plt.show()
```
![platform_sales](https://github.com/sangtvo/Video-Game-Sales-Queries/blob/main/images/platform_global_sales.png?raw=true)
* Playstation 2 seems to be the most popular console game in this era. However, as time goes on, PS titles slowly decline as new versions releases. One can notice that PS4 declined at least 4x as much as PS2. 
* In terms of portability, Nintendo DS is quite popular as it ranks 5th which means that consumers are more likely to play games on-the-go. 

### What are the top games that customers scored the highest?
```sql
q8 = '''
SELECT t.name, s.User_Score
FROM scores s
INNER JOIN titles t
    ON t.unique_id = s.unique_id
WHERE s.User_Score IS NOT NULL AND s.User_Score IS NOT 'tbd'
ORDER BY s.User_Score DESC, name ASC
LIMIT 10
'''
top_user_score = run_query(q8)
top_user_score
```
```python
# Convert variable in order to visualize the data.
top_user_score['Name'] = top_user_score['Name'].astype('category')
top_user_score['User_Score'] = top_user_score['User_Score'].astype('float')

ax7 = sns.barplot(x='User_Score', y='Name', data=top_user_score, palette='RdPu_r')
ax7.set_title('Best Titles by Users', size=14, weight='bold')
ax7.set_xlabel('User Score')
ax7.set_ylabel('Titles')
plt.show()
```
![best_titles](https://github.com/sangtvo/Video-Game-Sales-Queries/blob/main/images/best_titles_users.png?raw=true)
* The best title that has the highest user score rating is Breathe of Fire III with a score of 9.7. All other titles are fairly close to each other in the 9.4-9.6 range. 

Key Takeaways
---
* In 2008, publishers released a total of 1,427 titles which contributed majority of the sales in the data set. Perhaps this is the year when gaming became a life style and video game streamers were introduced. 
* North America contributed almost half of the sales which shows that this region has many gamers compared to other countries.
* Consumers have much higher expectations compared to critics who score the games. 
* The most popular portable gaming console title is the DS compared to the PSP. This is due to many fans who enjoy nintendo restricted titles that are not available on the PlayStation.
* Gamers prefer not to play any mind boggling or strategy genres and prefer more action-packed genres.