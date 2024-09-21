### fill the gaps below and send the script back

import pandas as pd
import sqlite3

DB_NAME = 'inventive_retail_group.db'

def create_sqlite_database(filename: str) -> None:
    """ create a database connection to an SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(filename)
        print(sqlite3.sqlite_version)
    except sqlite3.Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

create_sqlite_database(DB_NAME)
conn = sqlite3.connect(DB_NAME)

df = pd.read_csv('https://gist.githubusercontent.com/kevin336/acbb2271e66c10a5b73aacf82ca82784/raw/e38afe62e088394d61ed30884dd50a6826eee0a8/employees.csv')

employees = df.drop('SALARY', axis=1)
salaries = df[['EMPLOYEE_ID', 'SALARY']]

employees.to_sql('employees', conn, index=False, if_exists='replace')
salaries.to_sql('salaries', conn, index=False, if_exists='replace')

### find employees which managers work in other department
employees_working_in_other = pd.read_sql('''
SELECT e1.EMPLOYEE_ID, emp.FIRST_NAME, emp.LAST_NAME, emp.DEPARTMENT_ID, man.MANAGER_ID, man.DEPARTMENT_ID 
AS MANAGER_DEPARTMENT_ID
FROM employees emp
JOIN employees man ON e1.MANAGER_ID = man.EMPLOYEE_ID
WHERE emp.DEPARTMENT_ID != man.DEPARTMENT_ID;
''', conn)
print(employees_working_in_other)

### Calculate average salary for departments using window function
average_salaries_by_department = pd.read_sql('''
SELECT DEPARTMENT_ID, AVG(SALARY) OVER (PARTITION BY DEPARTMENT_ID) AS AVG_SALARY
FROM employees
JOIN salaries ON employees.EMPLOYEE_ID = salaries.EMPLOYEE_ID;
''', conn)
print(average_salaries_by_department)
