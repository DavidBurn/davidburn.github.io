---
layout: single
classes: wide
date: 2019-02-17
title: "Advent of code 2018 - Day 4"
excerpt: "A simple walkthrough of my solution"
header:
  overlay_image: "images/advent-doors.png"
  overlay_filter: 0.5
permalink: /advent-2018/day4/
sidebar:
  nav: advent-2018-sidebar
---

# Day 4
The challenge for day 4 was to find various methods to sneak past guards given their shift patterns and times they have fallen asleep.

[Day 4 instructions](https://adventofcode.com/2018/day/4)

I will run through my solution step by step, however I will leave the [full script at the end of this post](#script)

## Part 1
**Given the information about their previous shifts, which guard has been recorded asleep for the most minutes and which minute did that guard spend asleep most often?**



```python
import pandas as pd
import numpy as np
import re

with open('day4input.txt') as file:
    data = file.read().split('\n')
    
data[:10]
```




    ['[1518-09-14 00:54] wakes up',
     '[1518-04-15 23:58] Guard #373 begins shift',
     '[1518-07-25 00:53] wakes up',
     '[1518-07-04 00:45] wakes up',
     '[1518-07-26 00:51] wakes up',
     '[1518-06-21 00:43] falls asleep',
     '[1518-04-24 00:57] falls asleep',
     '[1518-11-20 00:52] wakes up',
     '[1518-04-20 00:39] falls asleep',
     '[1518-05-31 00:48] wakes up']



Once split by new lines, the input data is a list of unsorted strings. The majority of my solution revolves around creating a pandas dataframe and applying some functions to it upon creation to extract the information I need from these strings into new columns. I then solve for the final answer, the functions created for this purpose are detailed below.

The strings are first split on the ***\]*** to create a date string (column 0) and the remaining characters (column 1).


```python
def change_date(df):
    date = df[0].strip('[').replace('1518','2000')
    return pd.to_datetime(date)
```

***change_date*** strips the leading ***\[*** from the string, replaces the year 1518 for 2000 so the dates can convert into datetime (there must be a better workaround for this) and then returns the converted date.


```python
def get_mins(df):
    try:
        mins = df['diff'].total_seconds() / 60
        if 'wake' not in df[1]:
            mins = 0
        return int(mins)
    except AttributeError:
        pass
```

***get_mins*** takes the 'diff' column (created along with the dataframe, it is the difference between the date on that row and the previous row), extracts this value in seconds via the total_seconds() method and multiplies by 60 to return a minute value. If 'wake' is not present in column 1 this function returns 0, the idea being to get the number of minutes the guard was asleep by calculating the difference in time between a 'wake' row and its previous row ('falls asleep'). The try except clause just catches the attribute error for the first entry which doesn't have a 'diff' entry.


```python
def get_guard(df):
    m = re.search(r'\d+', df[1])
    if m:
        return int(m.group())
    return 0
```

***get_guard*** searches for any digits in the string in column 1, returing that number if any were found and 0 if not.


```python
df = (
    pd.DataFrame([x.split(']') for x in data])
    .drop(1267)
    .assign(date=lambda df: df.apply(change_date,axis=1))
    .sort_values(by='date')
    .assign(
        diff=lambda df: df['date'].diff(),
        minutes=lambda df: df.apply(get_mins, axis=1),
        guard=lambda df: df.apply(get_guard, axis=1),
    )
    .assign(guard=lambda df: df['guard'].replace(to_replace=0, method='ffill'))
)
```

The two other methods are ***sort_values*** which sorts the dataframe by date, and the final lambda function, this replaces all the 0 values for guard with the last occuring guard number.

Now we have our dataframe with all we need to solve the problem, our created columns are the four named ones.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>date</th>
      <th>diff</th>
      <th>minutes</th>
      <th>guard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>645</th>
      <td>[1518-01-12 23:57</td>
      <td>Guard #3209 begins shift</td>
      <td>2000-01-12 23:57:00</td>
      <td>NaT</td>
      <td>0</td>
      <td>3209</td>
    </tr>
    <tr>
      <th>1014</th>
      <td>[1518-01-13 00:13</td>
      <td>falls asleep</td>
      <td>2000-01-13 00:13:00</td>
      <td>00:16:00</td>
      <td>0</td>
      <td>3209</td>
    </tr>
    <tr>
      <th>602</th>
      <td>[1518-01-13 00:21</td>
      <td>wakes up</td>
      <td>2000-01-13 00:21:00</td>
      <td>00:08:00</td>
      <td>8</td>
      <td>3209</td>
    </tr>
    <tr>
      <th>387</th>
      <td>[1518-01-13 23:56</td>
      <td>Guard #751 begins shift</td>
      <td>2000-01-13 23:56:00</td>
      <td>23:35:00</td>
      <td>0</td>
      <td>751</td>
    </tr>
    <tr>
      <th>1190</th>
      <td>[1518-01-14 00:54</td>
      <td>falls asleep</td>
      <td>2000-01-14 00:54:00</td>
      <td>00:58:00</td>
      <td>0</td>
      <td>751</td>
    </tr>
  </tbody>
</table>
</div>



We first need to find the guard asleep for the most minutes in total. First we group the dataframe by guard and sum the minutes for each guard, using this summation to find the index of the top sleeping guard, then his guard number.


```python
# Group by guard
mins_per_guard = df.groupby('guard',as_index=False)['minutes'].sum()

#Take the top sleeping guards index
top_sleeper_idx = mins_per_guard['minutes'].idxmax()

# Find his guard number
top_sleeper = mins_per_guard['guard'][top_sleeper_idx]
```

To find the minute this guard was most asleep, I created a function ***minute_array*** which returns an empty array of zeros (representing each minute in the hour from 11pm to midnight) unless the row contains the word 'wake'. In the 'wake' rows the function returns an array of 0s and 1s with the 1s corresponding to the minutes the guard was asleep.


```python
def minute_array(row):
    empty = np.zeros(60)
    if 'wake' in row[1]:
        m = row['date'].minute
        lead = [0 for x in range(m-row['minutes'])]
        ones = [1 for x in range(row['minutes'])]
        post = [0 for x in range(60-len(lead)-len(ones))]
        return empty + (lead+ones+post)
    return empty
```

We then apply this function to each row of the dataframe filtered for only the top sleeping guards' entries, this list of arrays is then summed and the highest scoring minute obtained via the argmax method. The answer to part 1 can then be calculated (the number of guard most often asleep multiplied by the minute that guard was asleep most often).


```python
minute_arrays = df[df['guard']==top_sleeper].apply(minute_array, axis=1)

most_frequent = sum(minute_arrays).argmax()

print('Guard asleep most on the same minute :  {}'.format(top_sleeper))
print('On minute {}'.format(most_frequent))
print('Answer : {}'.format(top_sleeper * most_frequent))
```

    Guard asleep most on the same minute :  863
    On minute 46
    Answer : 39698


## Part 2 
**Of all the guards, find the guard who was asleep most often on one particular minute.**

To approach part 2 I again used the ***minute_array*** function, this time inside another function ***most_asleep_min***. This function again finds the minute a guard most frequently spends asleep, returning the total count of times spent asleep on that minute, the minute itself and the guard number.


```python
def most_asleep_min(x):
    minute_arrays = df[df['guard']==x].apply(minute_array, axis=1)
    most_freq = sum(minute_arrays).argmax()
    total = sum(minute_arrays).max()
    return total, most_freq, x
```

We can then iterate over all of the guard numbers, referring back to our summed *mins_per_guard* dataframe to retrive the guard numbers via a list comprehension. The answer for part 2 is the guard number multiplied by the value of the minute slept through most often.


```python
# Apply most_asleep_min to each guard number in the mins_per_guard dataframe
min_and_total = [most_asleep_min(x) for x in [x for x in mins_per_guard['guard']]]

# Store details of the guard with the most frequently slept minute
top = sorted(min_and_total, reverse=True)[0]

# Print answers
print('Guard number {} was asleep {} times on minute {}'.format(top[2],top[0],top[1]))
print('Answer : {}'.format(top[2]*top[1]))
```

    Guard number 373 was asleep 17.0 times on minute 40
    Answer : 14920


<a id="script" ></a>

## Full script




```python
import pandas as pd
import re
import numpy as np

with open('day4input.txt') as file:
    data = file.read().split('\n')
    
def change_date(row):
    date = row[0].strip('[').replace('1518','2000')
    return pd.to_datetime(date)

def get_mins(row):
    try:
        mins = row['diff'].total_seconds() / 60
        if 'wake' not in row[1]:
            mins = 0
        return int(mins)
    except AttributeError:
        pass
    
def get_guard(row):
    m = re.search(r'\d+', row[1])
    if m:
        return int(m.group())
    return 0

df = (
    pd.DataFrame([x.split(']') for x in data])
    .drop(1267)
    .assign(date=lambda df: df.apply(change_date,axis=1))
    .sort_values(by='date')
    .assign(
        diff=lambda df: df['date'].diff(),
        minutes=lambda df: df.apply(get_mins, axis=1),
        guard=lambda df: df.apply(get_guard, axis=1),
    )
    .assign(guard=lambda df: df['guard'].replace(to_replace=0, method='ffill'))
)
    
mins_per_guard = df.groupby('guard',as_index=False)['minutes'].sum()
top_sleeper_idx = mins_per_guard['minutes'].idxmax()
top_sleeper = mins_per_guard['guard'][top_sleeper_idx]

def minute_array(row):
    empty = np.zeros(60)
    if 'wake' in row[1]:
        m = row['date'].minute
        lead = [0 for x in range(m-row['minutes'])]
        ones = [1 for x in range(row['minutes'])]
        post = [0 for x in range(60-len(lead)-len(ones))]
        return empty + (lead+ones+post)
    return empty

minute_arrays = df[df['guard']==top_sleeper].apply(minute_array, axis=1)

most_frequent = sum(minute_arrays).argmax()

print('Guard asleep most on the same minute :  {}'.format(top_sleeper))
print('On minute {}'.format(most_frequent))
print('Answer : {}'.format(top_sleeper * most_frequent))

def most_asleep_min(x):
    minute_arrays = df[df['guard']==x].apply(minute_array, axis=1)
    most_freq = sum(minute_arrays).argmax()
    total = sum(minute_arrays).max()
    return total, most_freq, x

min_and_total = [most_asleep_min(x) for x in [x for x in mins_per_guard['guard']]]
top = sorted(min_and_total, reverse=True)[0]

print('Guard number {} was asleep {} times on minute {}'.format(top[2],top[0],top[1]))
print('Answer : {}'.format(top[2]*top[1]))
```

    Guard asleep most on the same minute :  863
    On minute 46
    Answer : 39698
    Guard number 373 was asleep 17.0 times on minute 40
    Answer : 14920

