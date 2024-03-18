---
title:  "S&P 500 stock - Descriptive analysis with Spark"
date:   2023-09-11 12:00:00 -500
categories: [Blog, Notebooks]
tags: [EDA, spark, big data, data science, análise de dados, python]
layout: post
comments: true
---



**Note:** All code is available on [Github](https://github.com/gallileugenesis/s-p-500-stock-descriptive-analysis-with-spark)

This notebook performs a descriptive analysis of [S&P 500 stock data](https://www.kaggle.com/datasets/camnugent/sandp500) using Apache Spark. The analysis covers various aspects of the stock market, including stock performance, trading volumes, and volatility.

#### Overview

In this notebook, we explore the following aspects of the S&P 500 stock data:

- **Data Loading**: We use Apache Spark to load the S&P 500 stock data from a CSV file into a DataFrame.

- **Data Exploration**: We conduct a basic exploratory data analysis (EDA) to gain insights into the dataset. This includes examining summary statistics and visualizing trends.

- **Stock Specific Analysis**: We answer specific questions about individual stocks, such as Apple (AAPL), including the highest stock price, trading volumes, and more.

- **Market-Wide Analysis**: We analyze the entire dataset to determine the frequency of days where closing prices are higher than opening prices, the most traded stock, the least traded stock, and more.



```python
import numpy as np  
import pandas as pd  
!pip install pyspark
from pyspark.sql import SparkSession

from pyspark.sql.functions import col, length
```

    Requirement already satisfied: pyspark in /opt/conda/lib/python3.10/site-packages (3.4.1)
    Requirement already satisfied: py4j==0.10.9.7 in /opt/conda/lib/python3.10/site-packages (from pyspark) (0.10.9.7)
    

#### Create a Spark session


```python
# Create a Spark session
spark = SparkSession.builder \
    .appName("S&P500StockAnalysis") \
    .getOrCreate()

```

#### Read the CSV file into a Spark DataFrame


```python
# Define the path to the CSV file
csv_file_path = "/kaggle/input/sandp500/all_stocks_5yr.csv"

# Read the CSV file into a Spark DataFrame
df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
```

                                                                                    

#### Show the first few rows of the DataFrame



```python
print(f'# rows: {df.count()}')
df.show()
```

    # rows: 619040
    +----------+-----+-----+-----+-----+--------+----+
    |      date| open| high|  low|close|  volume|Name|
    +----------+-----+-----+-----+-----+--------+----+
    |2013-02-08|15.07|15.12|14.63|14.75| 8407500| AAL|
    |2013-02-11|14.89|15.01|14.26|14.46| 8882000| AAL|
    |2013-02-12|14.45|14.51| 14.1|14.27| 8126000| AAL|
    |2013-02-13| 14.3|14.94|14.25|14.66|10259500| AAL|
    |2013-02-14|14.94|14.96|13.16|13.99|31879900| AAL|
    |2013-02-15|13.93|14.61|13.93| 14.5|15628000| AAL|
    |2013-02-19|14.33|14.56|14.08|14.26|11354400| AAL|
    |2013-02-20|14.17|14.26|13.15|13.33|14725200| AAL|
    |2013-02-21|13.62|13.95| 12.9|13.37|11922100| AAL|
    |2013-02-22|13.57| 13.6|13.21|13.57| 6071400| AAL|
    |2013-02-25| 13.6|13.76| 13.0|13.02| 7186400| AAL|
    |2013-02-26|13.14|13.42| 12.7|13.26| 9419000| AAL|
    |2013-02-27|13.28|13.62|13.18|13.41| 7390500| AAL|
    |2013-02-28|13.49|13.63|13.39|13.43| 6143600| AAL|
    |2013-03-01|13.37|13.95|13.32|13.61| 7376800| AAL|
    |2013-03-04| 13.5|14.07|13.47| 13.9| 8174800| AAL|
    |2013-03-05|14.01|14.05|13.71|14.05| 7676100| AAL|
    |2013-03-06|14.52|14.68|14.25|14.57|13243200| AAL|
    |2013-03-07| 14.7|14.93| 14.5|14.82| 9125300| AAL|
    |2013-03-08|14.99| 15.2|14.84|14.92|10593700| AAL|
    +----------+-----+-----+-----+-----+--------+----+
    only showing top 20 rows
    
    

#### Show basic statistics


```python
df.describe().show()
```

    [Stage 129:============================>                            (2 + 2) / 4]

    +-------+----------------+-----------------+-----------------+-----------------+-----------------+------+
    |summary|            open|             high|              low|            close|           volume|  Name|
    +-------+----------------+-----------------+-----------------+-----------------+-----------------+------+
    |  count|          619029|           619032|           619032|           619040|           619040|619040|
    |   mean|83.0233343145481|83.77831069347307|82.25609641375267|83.04376276476548|4321823.395568945|  null|
    | stddev|97.3787690433234|98.20751890446383|96.50742105809076|97.38974800165785| 8693609.51196759|  null|
    |    min|            1.62|             1.69|              1.5|             1.59|                0|     A|
    |    max|          2044.0|          2067.99|          2035.11|           2049.0|        618237630|   ZTS|
    +-------+----------------+-----------------+-----------------+-----------------+-----------------+------+
    
    

                                                                                    

#### How many records are there in the database for Apple (AAPL) stock?


```python
# Count the occurrences of "AAPL"
count_aapl = df.filter(df["Name"] == "AAPL").count()
print(f"There are {count_aapl} records for AAPL stock")
```

    There are 1259 records for AAPL stock
    

#### How many different companies have in the database?


```python
# Get distinct or unique values in the "Name" column
number_of_unique_stocks  = df.select("Name").distinct().count()
print(f"There are {number_of_unique_stocks} distinct stocks in the database")
```

    There are 505 distinct stocks in the database
    

#### How often is the closing price of a stock higher than the opening price?



```python
close_greater_open = df.filter(df["Close"] > df["Open"]).count()
freq_close_greater_open = 100*close_greater_open/df.count()
print(f"The closing prices are higher than the opening prices in {freq_close_greater_open:.2f}% of cases")
```

    The closing prices are higher than the opening prices in 51.53% of cases
    

#### What is the highest value of Apple (AAPL) stocks in history?



```python
aapl = df.filter(df["Name"] == "AAPL")
biggest_aapl_value = aapl.agg({"high": "max"}).collect()[0][0]
print(f"The highest value of Apple (AAPL) stocks in history was $ {biggest_aapl_value}")
```

    The highest value of Apple (AAPL) stocks in history was $ 180.1
    

#### Which stock has the highest volatility?
One way to measure this is by calculating the standard deviation of the closing price for each stock and identifying the stock with the highest standard deviation.


```python
greatest_std = df.groupby("Name").agg({"close": "std"}).orderBy(col("stddev(close)").desc()).first()
print(f"The stock with the highest volatility was {greatest_std['Name']}, with a standard deviation in closing price of $ {greatest_std['stddev(close)']:.2f}")

```

    The stock with the highest volatility was PCLN, with a standard deviation in closing price of $ 320.53
    

#### What is the day with the highest total trading volume on the stock exchange?



```python
greatest_volume_date = df.groupby("date").agg({"volume": "sum"}).orderBy(col("sum(volume)").desc()).first()
print(f"The day with the highest trading volume on the stock exchange was {greatest_volume_date['date']}, with a total of $ {greatest_volume_date['sum(volume)']:.2f}")
```

    The day with the highest trading volume on the stock exchange was 2015-08-24, with a total of $ 4607945196.00
    

                                                                                    

#### Which stock is the most traded on the stock exchange in terms of transaction volume?



```python
most_traded_stock = df.groupby("Name").agg({"volume": "sum"}).orderBy(col("sum(volume)").desc()).first()
print(f"The most traded stock on the stock exchange, in terms of transaction volume, was {most_traded_stock['Name']}, with a total of $ {most_traded_stock['sum(volume)']:.2f}")
```

    The most traded stock on the stock exchange, in terms of transaction volume, was BAC, with a total of $ 117884953591.00
    

#### How many stocks start with the letter “A”?




```python
starts_with_A = df.select("Name").distinct().filter(col("Name").startswith("A")).count()
print(f"{starts_with_A} stocks start with the letter A")
```

    59 stocks start with the letter A
    

#### How often is the highest price of the day also the closing price?



```python
high_equals_close = df.filter(df["high"] == df["close"]).count()
freq_high_equals_close = 100*high_equals_close/df.count()
print(f"The highest price of the day is also the closing price in {freq_high_equals_close:.2f}% of cases")
```

    The highest price of the day is also the closing price in 1.20% of cases
    

#### On which day did Apple stock rise the most between opening and closing, in absolute terms?



```python
aapl_exp = aapl.withColumn('diff_open_close', aapl['open'] - aapl['close'])
max_abs_diff_day = aapl_exp.select("date", "diff_open_close").orderBy(col("diff_open_close").desc()).first()
print(f"Apple stock rose the most between opening and closing, in absolute terms, on {max_abs_diff_day['date']}")
```

    Apple stock rose the most between opening and closing, in absolute terms, on 2015-08-25
    

                                                                                    

#### On average, what is the daily trading volume of AAPL stocks?



```python
avg_volume_aapl = aapl.agg({"volume": "mean"}).collect()[0][0]
print(f"The average daily trading volume of AAPL stocks is $ {avg_volume_aapl:.2f}")
```

    The average daily trading volume of AAPL stocks is $ 54047899.74
    

#### How many stocks have 1, 2, 3, 4, and 5 characters in their names, respectively?



```python
name_df = df.groupby("Name").count()
# Adicione uma coluna para calcular o comprimento do nome
name_df = name_df.withColumn("name_length", length(col("Name")))
groupby_name_length = name_df.groupby("name_length").count().\
                                        orderBy(col("name_length").asc()).show()
groupby_name_length
```

    +-----------+-----+
    |name_length|count|
    +-----------+-----+
    |          1|   10|
    |          2|   50|
    |          3|  323|
    |          4|  117|
    |          5|    5|
    +-----------+-----+
    
    

#### Which stock is the least traded on the stock exchange, in terms of transaction volume?



```python
least_traded_stock = df.groupby("Name").agg({"volume": "sum"}).orderBy(col("sum(volume)").asc()).first()
print(f"The least traded stock on the stock exchange was {least_traded_stock['Name']}, with a total volume of $ {least_traded_stock['sum(volume)']:.2f}")
```

    The least traded stock on the stock exchange was APTV, with a total volume of $ 92947779.00
    

#### Stop spark session


```python
spark.stop()
```

**Note:** All code is available on [Github](https://github.com/gallileugenesis/s-p-500-stock-descriptive-analysis-with-spark)