![Pandas10minutes](https://socialify.git.ci/julioaranajr/Pandas_10_minutes_to_Go/image?description=1&forks=1&issues=1&language=1&name=1&owner=1&pattern=Solid&pulls=1&stargazers=1&theme=Dark)


---

# Pandas 10 minutes to Go

This is a quick guide to get started with Pandas. It is based on the official Pandas documentation. The original documentation can be found in [Getting started](https://pandas.pydata.org/pandas-docs/stable/getting_started/index.html).

## Getting Started with Pandas

- [Pandas 10 minutes to Go](#pandas-10-minutes-to-go)
  - [Getting Started with Pandas](#getting-started-with-pandas)
  - [Installation](#installation)
  - [Pandas Data Structures](#pandas-data-structures)
    - [{ pd.Series }](#-pdseries-)
    - [{ pd.date\_range }](#-pddate_range-)
    - [{ pd.DataFrame }](#-pddataframe-)
  - [Viewing Data](#viewing-data)
    - [{ df.head() }](#-dfhead-)
    - [{ df.tail() }](#-dftail-)
    - [{ df.index }](#-dfindex-)
    - [{ df.columns }](#-dfcolumns-)
    - [{ df.values }](#-dfvalues-)
    - [{ df.describe() }](#-dfdescribe-)
    - [{ df.T }](#-dft-)
    - [{ df.sort\_index(axis=1, ascending=False) }](#-dfsort_indexaxis1-ascendingfalse-)
    - [{ df.sort\_values(by='B') }](#-dfsort_valuesbyb-)
  - [Selection](#selection)
    - [{ df.loc }](#-dfloc-)
    - [{ df.iloc }](#-dfiloc-)
  - [Missing Data](#missing-data)
    - [{ df.reindex }](#-dfreindex-)
    - [{ df.dropna }](#-dfdropna-)
    - [{ df.fillna }](#-dffillna-)
    - [{ pd.isna }](#-pdisna-)
  - [Operations](#operations)
    - [{ df.mean }](#-dfmean-)
    - [{ df.sub }](#-dfsub-)
    - [{ df.apply }](#-dfapply-)
  - [Merge](#merge)
    - [{ pd.concat }](#-pdconcat-)
  - [Grouping](#grouping)
    - [{ df.groupby }](#-dfgroupby-)
  - [Reshaping](#reshaping)
    - [{ df.stack }](#-dfstack-)
  - [Time Series](#time-series)
    - [{ ts.resample }](#-tsresample-)
  - [Categoricals](#categoricals)
    - [{ df\["grade"\] = df\["raw\_grade"\].astype("category") }](#-dfgrade--dfraw_gradeastypecategory-)
  - [Plotting](#plotting)
    - [{ ts.plot }](#-tsplot-)
  - [Getting Data In/Out](#getting-data-inout)
    - [{ df.to\_csv('foo.csv') }](#-dfto_csvfoocsv-)
    - [{ pd.read\_csv('foo.csv') }](#-pdread_csvfoocsv-)
    - [{ df.to\_excel('foo.xlsx', sheet\_name='Sheet1') }](#-dfto_excelfooxlsx-sheet_namesheet1-)
    - [{ pd.read\_excel('foo.xlsx', 'Sheet1', index\_col=None, na\_values=\['NA'\]) }](#-pdread_excelfooxlsx-sheet1-index_colnone-na_valuesna-)
    - [{ df.to\_hdf('foo.h5', 'df') }](#-dfto_hdffooh5-df-)
    - [{ pd.read\_hdf('foo.h5', 'df') }](#-pdread_hdffooh5-df-)
    - [{ df.to\_sql('table', conn) }](#-dfto_sqltable-conn-)
    - [{ pd.read\_sql('SELECT \* FROM table', conn) }](#-pdread_sqlselect--from-table-conn-)
    - [{ df.to\_json('foo.json') }](#-dfto_jsonfoojson-)
    - [{ pd.read\_json('foo.json') }](#-pdread_jsonfoojson-)
    - [{ df.to\_html('foo.html') }](#-dfto_htmlfoohtml-)
    - [{ df.to\_clipboard() }](#-dfto_clipboard-)
  - [Gotchas](#gotchas)
  - [Conclusion](#conclusion)
  - [Next Tutorial](#next-tutorial)

## Installation

To install Pandas, you can use the following command:

```bash
pip install pandas
```

## Pandas Data Structures

Pandas introduces two new data structures to Python: Series and DataFrame.

You can think of Series as a one-dimensional array and DataFrame as a two-dimensional array.

### { pd.Series }

A Series is a one-dimensional array with labels.

You can create a Series by passing a list of values, letting Pandas create a default integer index.

```python
import pandas as pd
import numpy as np

s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)
```

### { pd.date_range }

A date_range is a method to create a range of dates. it is similar to SQL's DATEADD.

```python
dates = pd.date_range('20130101', periods=6)
print(dates)
```

### { pd.DataFrame }

A DataFrame is a two-dimensional array with labels. You can create a DataFrame by passing a NumPy array with a datetime index and labeled columns. it is similar to SQL's CREATE TABLE.

```python
dates = pd.date_range('20130101', periods=6)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)
```

**REMEMBER:**

- Import the package, aka import pandas as pd

- A table of data is stored as a pandas DataFrame

- Each column in a DataFrame is a Series

- You can do things by applying a method to a DataFrame or Series

## Viewing Data

Here are some ways to view the data in Pandas:

### { df.head() }

df.head() returns the first n rows of the DataFrame. By default, n=5. it is similar to SQL's TOP.

```python
print(df.head())
```

### { df.tail() }

df.tail() returns the last n rows of the DataFrame. By default, n=5. it is similar to SQL's TOP.

```python
print(df.tail())
```

### { df.index }

df.index returns the index of the DataFrame. it is similar to SQL's INDEX.

```python
print(df.index)
```

### { df.columns }

df.columns returns the columns of the DataFrame. it is similar to SQL's COLUMN_NAME.

```python
print(df.columns)
```

### { df.values }

df.values returns the values of the DataFrame. it is similar to SQL's VALUES.

```python
print(df.values)
```

### { df.describe() }

df.describe() returns a summary of the DataFrame. it is similar to SQL's DESCRIBE.

```python
print(df.describe())
```

### { df.T }

df.T transposes the DataFrame. it is similar to SQL's TRANSPOSE.

```python
print(df.T)
```

### { df.sort_index(axis=1, ascending=False) }

df.sort_index(axis=1, ascending=False) sorts the DataFrame by index. it is similar to SQL's ORDER BY.

```python
print(df.sort_index(axis=1, ascending=False))
```

### { df.sort_values(by='B') }

df.sort_values(by='B') sorts the DataFrame by values. it is similar to SQL's ORDER BY.

```python
print(df.sort_values(by='B'))
```

## Selection

You can select data in Pandas using the following methods:

```python
print(df['A'])
print(df[0:3])
print(df['20130102':'20130104'])
```

You can also select data by label:

### { df.loc }

df.loc selects data by label. it is similar to SQL's WHERE.

```python
print(df.loc[dates[0]])
print(df.loc[:, ['A', 'B']])
print(df.loc['20130102':'20130104', ['A', 'B']])
print(df.loc['20130102', ['A', 'B']])
print(df.loc[dates[0], 'A'])
```

You can also select data by position:

### { df.iloc }

df.iloc selects data by position. it is similar to SQL's WHERE.

```python
print(df.iloc[3])
print(df.iloc[3:5, 0:2])
print(df.iloc[[1, 2, 4], [0, 2]])
print(df.iloc[1:3, :])
print(df.iloc[:, 1:3])
```

## Missing Data

Pandas provides several methods for handling missing data:

### { df.reindex }

df.reindex changes the index of the DataFrame. it is similar to SQL's REINDEX.

```python
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
print(df1)
```

### { df.dropna }

df.dropna drops any rows with missing data. it is similar to SQL's DELETE.

```python
print(df1.dropna(how='any'))
```

### { df.fillna }

df.fillna fills missing data. it is similar to SQL's COALESCE. it is similar to SQL's COALESCE.

```python
print(df1.fillna(value=5))
```

### { pd.isna }

pd.isna checks for missing data. it is similar to SQL's IS NULL. it is similar to SQL's IS NULL.

```python
print(pd.isna(df1))
```

## Operations

You can perform operations on data in Pandas:

### { df.mean }

df.mean() returns the mean of the data. it is similar to SQL's AVG.

```python
print(df.mean())
print(df.mean(1))
```

### { df.sub }

df.sub subtracts data. it is similar to SQL's EXCEPT.

```python
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
print(df.sub(s, axis='index'))
```

### { df.apply }

df.apply applies a function to the data. it is similar to SQL's APPLY.

```python
print(df.apply(np.cumsum))
print(df.apply(lambda x: x.max() - x.min()))
```

## Merge

You can merge data in Pandas:

### { pd.concat }

pd.concat concatenates data. it is similar to SQL's UNION.

```python
df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]
print(pd.concat(pieces))
```

## Grouping

You can group data in Pandas:

### { df.groupby }

df.groupby groups data. it is similar to SQL's GROUP BY.

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})
print(df)
print(df.groupby('A').sum())
print(df.groupby(['A', 'B']).sum())
```

## Reshaping

You can reshape data in Pandas:

### { df.stack }

df.stack stacks the DataFrame. it is similar to SQL's PIVOT.

```python
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]
print(df2)
stacked = df2.stack()
print(stacked)
print(stacked.unstack())
print(stacked.unstack(1))
print(stacked.unstack(0))
```

## Time Series

Pandas has simple tools for working with time series data:

### { ts.resample }

ts.resample resamples the time series data. it is similar to SQL's DATEADD.

```python
rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(ts.resample('5Min').sum())
```

## Categoricals

Pandas can include categorical data in a DataFrame:

### { df["grade"] = df["raw_grade"].astype("category") }

df["grade"] = df["raw_grade"].astype("category") converts the raw_grade column to a category. it is similar to SQL's CAST.

```python
df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
df["grade"] = df["raw_grade"].astype("category")
print(df["grade"])
```

## Plotting

You can plot data in Pandas:

### { ts.plot }

ts.plot plots the data. it is similar to SQL's PLOT.
ts.cumsum() returns the cumulative sum of the data.

```python
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
```

## Getting Data In/Out

Pandas can read and write data in various formats:

### { df.to_csv('foo.csv') }

df.to_csv('foo.csv') writes the DataFrame to a CSV file. it is similar to SQL's BCP.

```python
df.to_csv('foo.csv')
```

### { pd.read_csv('foo.csv') }

pd.read_csv('foo.csv') reads the DataFrame from a CSV file. it is similar to SQL's BCP.

```python
pd.read_csv('foo.csv')
```

### { df.to_excel('foo.xlsx', sheet_name='Sheet1') }

df.to_excel('foo.xlsx', sheet_name='Sheet1') writes the DataFrame to an Excel file. it is similar to SQL's BCP.

```python
df.to_excel('foo.xlsx', sheet_name='Sheet1')
```

### { pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA']) }

pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA']) reads the DataFrame from an Excel file. it is similar to SQL's BCP.

```python
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
```

### { df.to_hdf('foo.h5', 'df') }

df.to_hdf('foo.h5', 'df') writes the DataFrame to an HDF5 file. it is similar to SQL's BCP.

```python
df.to_hdf('foo.h5', 'df')
```

### { pd.read_hdf('foo.h5', 'df') }

pd.read_hdf('foo.h5', 'df') reads the DataFrame from an HDF5 file. it is similar to SQL's BCP.

```python
pd.read_hdf('foo.h5', 'df')
```

### { df.to_sql('table', conn) }

df.to_sql('table', conn) writes the DataFrame to a SQL database. it is similar to SQL's BCP.

```python
df.to_sql('table', conn)
```

### { pd.read_sql('SELECT \* FROM table', conn) }

pd.read_sql('SELECT \* FROM table', conn) reads the DataFrame from a SQL database. it is similar to SQL's BCP.

```python
pd.read_sql('SELECT * FROM table', conn)
```

### { df.to_json('foo.json') }

df.to_json('foo.json') writes the DataFrame to a JSON file. it is similar to SQL's BCP.

```python
df.to_json('foo.json')
```

### { pd.read_json('foo.json') }

pd.read_json('foo.json') reads the DataFrame from a JSON file. it is similar to SQL's BCP.

```python
pd.read_json('foo.json')
```

### { df.to_html('foo.html') }

df.to_html('foo.html') writes the DataFrame to an HTML file. it is similar to SQL's BCP.

```python
df.to_html('foo.html')
```

You can also copy data to the clipboard:

### { df.to_clipboard() }

df.to_clipboard() copies the DataFrame to the clipboard. it is similar to SQL's BCP.

```python
df.to_clipboard()
```

## Gotchas

Here are some common gotchas in Pandas:

If you try to evaluate a Series in an if statement, you will get an error:

```python
if pd.Series([False, True, False]):
    print("I was true")
```

TO FIX THIS, you can use any() or all() to evaluate the Series:

```python
if pd.Series([False, True, False]).any():
    print("I was true")
```

## Conclusion

With this guide, you should be able to get started with Pandas. Check out how generate fake data with this [Tutorial generating-fake-data](/home/dci-student/DCI/0_Repositories/2409/Pandas_10_minutes_to_Go/generating-fake-data.ipynb). We wil be emulating some of the free datasets from [Kaggle](https://www.kaggle.com/datasets) in particular the Netflix original films IMDB score to generate something similar.

## Next Tutorial

In the next tutorial, we will be making and introduction to [Polars](https://docs.pola.rs/), a new way to manage your notes and knowledge. Stay tuned!

Happy coding! 🚀
