---
layout: single
classes: wide
date: 2019-04-11
title: "Advent of code 2018 - Day 6"
excerpt: "A simple walkthrough of my solution"
header:
  overlay_image: "images/advent-doors.png"
  overlay_filter: 0.5
permalink: /advent-2018/day6/
sidebar:
  nav: advent-2018-sidebar
---

# Day 6 : Chronal Coordinates
Input for day 6 is a list of coordinates, the instructions are best read in full but involve calculating the [Manhattan distance](https://en.wikipedia.org/wiki/Taxicab_geometry) from each coordinate to each other point.

[Day 6 instructions](https://adventofcode.com/2018/day/6)

Step by step walkthrough of my solution follows, the full script can be found at [the end of this post](#script)

## Part 1
**Using only the Manhattan distance, determine the area around each coordinate by counting the number of integer X,Y locations that are closest to that coordinate (and aren't tied in distance to any other coordinate).**

**Your goal is to find the size of the largest area that isn't infinite.**


```python
import numpy as np

with open('day6input.txt') as file:
    coords = file.read()
    
coords[:30]
```




    '63, 142\n190, 296\n132, 194\n135,'



The coordinates need to be split and cleaned. Splitting on whitespace leaves a list of strings of format [x-coord, y-coord, x-coord, y-coord], this can be cleaned up by removing the commas and pairing each x-coordinate with it's corresponding y-coordinate.


```python
coords = coords.split()

x_coords = [int(coords[i].strip(',')) for i in range(0,len(coords),2)]
y_coords = [int(coords[i]) for i in range(1, len(coords),2)]

coords = list(zip(x_coords, y_coords))
```


```python
coords[:10]
```




    [(63, 142),
     (190, 296),
     (132, 194),
     (135, 197),
     (327, 292),
     (144, 174),
     (103, 173),
     (141, 317),
     (265, 58),
     (344, 50)]



To find which of our coordinates is closest to each other point in the whole X,Y plane I create the function  *get_shortest_distance*. The Manhattan distance between two points is simply the absolute difference between the two x coordinates plus the absolute difference between the two y coordinates. Our function calculates the manhattan distance between a given point and each coordinate in our *coords* list, storing these distances in an array.

The function then returns -1 if the closest of our coordinates is tied in distance to the next closest. If the closest coordinate is unique then the function returns the position of the coordinates in our *coords* list.


```python
def get_shortest_distance(point):
    distances = [abs(point[0] - coord[0]) + abs(point[1] - coord[1]) for coord in coords]
    distances = np.array(distances)
    if sorted(distances)[0] == sorted(distances)[1]:
        return -1
    return distances.argmin() + 1
```

I then created a list of all the coordinates using itertools, getting all permutations of coordinates from 0 to the maximum value in our *coords* list. Then our function can be used to find which coordinate is closest to each point in the permutation list. 


```python
import itertools

largest_coord = max(x_coords+y_coords)
# Create permutations of all coordinates
perm = itertools.product(list(range(largest_coord)),repeat=2)

# Store closest coordinate (or -1 if tied)
distances = [get_shortest_distance(x) for x in perm]
```

This list of distances can be visualised by reshaping it into a grid and plotting it as a colormap. The areas around the edges of the plot are the infinite areas for the purposes of the task and do not count when determining the size of the largest area.


```python
import matplotlib.pyplot as plt

grid = np.array(distances).reshape(largest_coord, largest_coord)

plt.figure(figsize=(12,8))
plt.title('Visualisation of areas')
plt.scatter(x_coords, y_coords, color='r', label='Coordinates')
plt.legend()
plt.imshow(grid.T)
plt.axis('off')
```




    (-0.5, 353.5, 353.5, -0.5)




![png](/images/day6.png)


To find the largest inner area we must first collate all of the infinte areas (any area at the edge of the map)


```python
edges = [grid[0], grid[max(x_coords)-1], grid[:,0], grid[:, max(y_coords)-1]]

np.unique(edges)
```




    array([-1,  1,  2,  8,  9, 10, 12, 13, 14, 15, 17, 19, 20, 23, 25, 28, 29,
           30, 33, 34, 35, 36, 37, 43, 44, 45, 48])



All of these values correspond to coordinates in our coords list and to infinite areas on the colour map. Therefore we can simply calculate the area of all other areas and the highest of these will be that of the largest inner area.


```python
unique, count = np.unique(grid, return_counts=True)
inner_areas = [x for x in sorted(zip(count, unique)) if x[1] not in np.unique(edges)]
```


```python
inner_areas[-1]
```




    (5429, 50)



The largest inner area is 5249, coordinate number 50 on the list
## Part 2 
#### What is the size of the region containing all locations which have a total distance to all given coordinates of less than 10000?

We already have most of the work done for this problem, I created a new function which sums the distance between a point and each coordinate. If the sum is less than 10000 the function returns a 1, if not it returns a 0.


```python
def is_point_under_total_distance(point, distance=10000):
    distances = [abs(point[0] - coord[0]) + abs(point[1] - coord[1]) for coord in coords]
    if sum(distances) < 10000:
        return 1
    return 0
```

I completed the next part in two ways, the first method fulfils a functional programming challenge I was partaking in with a friend at the time, the second method was my first method of solving the problem


```python
# Method 1

# Create list of coordinates, run a list comprehension using our function and summing the 1/0 values

p = itertools.product(list(range(max(x_coords+y_coords))),repeat=2)
under_10k = [is_point_under_total_distance(x) for x in p]
print(sum(under_10k))
```

    32614



```python
# Method 2

# Create grid of zeros, run each coordinate through our function and sum the total of the grid
grid2 = np.zeros((max(x_coords), max(y_coords)))
for i in range(grid2.shape[0]):
    for j in range(grid2.shape[1]):
        grid2[(i,j)] = is_point_under_total_distance((i,j))

print(grid2.sum())
```

    32614.0


<a id= "script" ></a>

## Full script


```python
import numpy as np

with open('day6input.txt') as file:
    coords = file.read().split()

x_coords = [int(coords[i].strip(',')) for i in range(0,len(coords),2)]
y_coords = [int(coords[i]) for i in range(1, len(coords),2)]

COORDS= list(zip(x_coords, y_coords))

grid = np.zeros((max(x_coords), max(y_coords)))

def get_shortest_distance(point,coords=COORDS):
    distances = [abs(point[0] - coord[0]) + abs(point[1] - coord[1]) for coord in coords]
    distances = np.array(distances)
    if sorted(distances)[0] == sorted(distances)[1]:
        return -1
    return distances.argmin() + 1
    
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        grid[(i,j)] = get_shortest_distance((i,j))

edges = [grid[0], grid[max(x_coords)-1], grid[:,0], grid[:, max(y_coords)-1]]

flattened_edges = [y for x in edges for y in x]
unique, count = np.unique(grid, return_counts=True)
inner_areas = [x for x in sorted(zip(count, unique)) if x[1] not in flattened_edges]

print(inner_areas[-1])

"""
Part 2 - Find all areas with less than 10000 total distance from all points
"""

def is_point_under_total_distance(point, coords = COORDS, distance=10000):
    distances = [abs(point[0] - coord[0]) + abs(point[1] - coord[1]) for coord in coords]
    if sum(distances) < 10000:
        return 1
    return 0

grid2 = np.zeros((max(x_coords), max(y_coords)))
for i in range(grid2.shape[0]):
    for j in range(grid2.shape[1]):
        grid2[(i,j)] = is_point_under_total_distance((i,j))

print(grid2.sum())


```

    (5429, 50.0)
    32614.0

