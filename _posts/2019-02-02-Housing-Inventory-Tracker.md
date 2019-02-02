---
layout: single
classes: wide
date: 2019-02-02
title: "Housing Inventory Tracker"
header:
  overlay_image: "images/housing-inventory.jpg"
  overlay_filter: 0.5
tags: [challenges, python]
categories: [functional programming]
excerpt: "A PyBites code challenge"
---

# Housing Inventory Tracker - A PyBites code challenge

I have recently stumbled across the twitter tag #100daysOfCode, the idea being to code or do some supplementary learning towards coding for 100 days in a row. I am 7 days in and am already finding myself thinking about code and problems I could solve much more regularly throughout my daily life.

Todays challenge was taken from [PyBites Code Challenges](https://pybit.es/pages/challenges.html), a blog hosting free, open ended tasks released periodically to complete in your own time and then share with the community. This is task 8, a good year behind the most current challenge but I hope to catch up during my 100 days of code. 

## The challenge

The challenge is simple with some added options if you wish to take it a few steps further. The instructions from PyBites:

#### House Inventory Tracker Requirements

- Create a list of rooms (doesn't have to use the list type).

- Each room in your rooms list needs to contain at least 5 items (ie, TV, couch, table, etc) and the relative dollar value of each item.

- The script you will write will print out each room along with the individual items and values. This needs to be properly formatted, eg: no printing a dict as is.

There were also some bonus parts to the challenge:

#### These are bonus features. Not required but cool to try if you're interested:

- Create some sort of program shell with a menu system around this. ie, "Which room would you like to list out?"

- If you're really game, allow users to create rooms and update information.

- You could even make an API with Flask or your preferred framework

- Print the total dollar value of each room and the entire house.

- Have persistent storage of the data. sqlite3 = stdlib and light-weight, but feel free to use your preferred DB / module.

There are clearly many ways in which to approach this challenge, simply entering the information in a script in the form of lists or a dataframe would allow the results to be printed to the screen. However having had little practice creating classes I opted to model the house as a class instance, allowing for the whole process to be easily user controlled after the script has been run, first creating the house, then adding rooms and items. 


```python
import json

class House:
    """
    A class for storing household inventory.
    
    Attributes
        name (str) : Name of the house.
        total_value (int) : Total value of items in the house.
        rooms (dict) : A dictionary of rooms, each rooms value being a dictionary of items
                       within that room along with its dollar value.
    """
    def __init__(self, name, rooms = None):
        """
        The constructor for House class.
        
        Parameters:
            name (str) : Name of the house.
            rooms (dict) : Defaults to None, an optional input allowed a previously created
                           House class to be loaded.
        
        """
        self.name = name
        self.total_value = 0
        if rooms:
            self.rooms = rooms
            self.update_total()
        else:
            self.rooms = {}
    
    def update_total(self):
        "Updates total_value attribute"
        self.total_value = sum(sum(self.rooms[room].values()) for room in self.rooms.keys())
    
    def add_room(self, name):
        "Add room name to rooms attribute"
        self.rooms[name] = {}
        
    def add_item(self, item_name, value, room):
        """
        Add item to rooms attribute, creating room if room does not exist.
        
        Parameters:
            item_name (str, list) : Item or items to be added to house inventory
            value (str, list) : Value of item or items to be added.
            room (str) : Location of item or items to be added.
        
        tota_value updates after items are added.
        """
        if room not in house.rooms:
            self.add_room(room)
        if type(item_name) == list:
            for item, value in zip(item_name, value):
                self.rooms[room][item] = value
        else:
            self.rooms[room][item_name] = value
        self.update_total()
        
    def remove_item(self, item_name, room):
        """
        Remove item from rooms attribute.
        
        Parameters:
            item_name (str) : Name of item to remove.
            room (str) : Location of item to remove
            
        total_value updates after items are removed.
        """
        del self.rooms[room][item_name]
        self.update_total()
    
    def print_items(self):
        "Print full inventory including room totals and grand total."
        print('{:*^40}'.format(self.name))
        for room in self.rooms.keys():
            print(room)
            for item, value in self.rooms[room].items():
                print('    {:<15} : {}'.format(item, value))
            print('Room total : {}'.format(sum(self.rooms[room].values())))
            print('-'*10)
        print('Grand Total : {}'.format(self.total_value))
        
    def save_house_to_disk(self, path='house.json'):
        """
        Save house class to disk as JSON file."
        
        Parameters:
            path (str) : Filepath for outfile.
        """
        with open(path, 'w') as file:
            json.dump(self.rooms, file)

```

As defined in the docstrings, the class has three attributes, a name, total value and "rooms" which holds the full inventory and each items location. The methods are all self explanatory, the update_total is used to update the *total_value* attribute when items are added or removed from the inventory. 

We can now create a house and add some items.


```python
house = House('My Place')
house.add_room('Garage')
house.rooms
```




    {'Garage': {}}




```python
house.add_item('Bike', 500, 'Garage')
house.rooms
```




    {'Garage': {'Bike': 500}}




```python
things = ['Saw', 'Roof Box', 'Sofa', 'Workbench']
values = [200, 250, 150, 25]

house.add_item(things, values, 'Garage')
house.rooms
```




    {'Garage': {'Bike': 500,
      'Roof Box': 250,
      'Saw': 200,
      'Sofa': 150,
      'Workbench': 25}}




```python
house.print_items()
```

    ****************My Place****************
    Garage
        Bike            : 500
        Saw             : 200
        Roof Box        : 250
        Sofa            : 150
        Workbench       : 25
    Room total : 1125
    ----------
    Grand Total : 1125



```python
house.remove_item('Saw', 'Garage')
house.print_items()
```

    ****************My Place****************
    Garage
        Bike            : 500
        Roof Box        : 250
        Sofa            : 150
        Workbench       : 25
    Room total : 925
    ----------
    Grand Total : 925



```python
dr_items = ['Dining Table', 'Sideboard', 'Lighting', 'Cutlery', 'Placemats', 'Curtains']
dr_values = [700, 300, 100, 25, 20, 90]

house.add_item(dr_items, dr_values, 'Dining Room')

lr_things = ['TV', 'TV Stand', 'Coffee Table', 'Sofa', 'Curtains']
lr_values = [1000, 300, 150, 200, 100]

house.add_item(lr_things, lr_values, 'Living Rooms')

house.add_item('Shelving', 25, 'Garage')

house.print_items()
```

    ****************My Place****************
    Garage
        Bike            : 500
        Roof Box        : 250
        Sofa            : 150
        Workbench       : 25
        Shelving        : 25
    Room total : 950
    ----------
    Dining Room
        Dining Table    : 700
        Sideboard       : 300
        Lighting        : 100
        Cutlery         : 25
        Placemats       : 20
        Curtains        : 90
    Room total : 1235
    ----------
    Living Rooms
        TV              : 1000
        TV Stand        : 300
        Coffee Table    : 150
        Sofa            : 200
        Curtains        : 100
    Room total : 1750
    ----------
    Grand Total : 3935


Now we have finished entering our items, we can call the *save_house_to_disk* method, leaving the default filepath 'house.json' and save our house for later. 


```python
house.save_house_to_disk()
```

To load the house, we add a supplementary function. Provided we navigate to the same working directory we last worked on the house, we can use the default path again.


```python
def load_house(name, path='house.json'):
    with open(path) as file:
        house_data = json.load(file)
    house = House(name, house_data)
    return house

loaded_house = load_house('House reloaded')
```


```python
loaded_house.print_items()
```

    *************House reloaded*************
    Garage
        Bike            : 500
        Roof Box        : 250
        Sofa            : 150
        Workbench       : 25
        Shelving        : 25
    Room total : 950
    ----------
    Dining Room
        Dining Table    : 700
        Sideboard       : 300
        Lighting        : 100
        Cutlery         : 25
        Placemats       : 20
        Curtains        : 90
    Room total : 1235
    ----------
    Living Rooms
        TV              : 1000
        TV Stand        : 300
        Coffee Table    : 150
        Sofa            : 200
        Curtains        : 100
    Room total : 1750
    ----------
    Grand Total : 3935


## Summary

A simple house inventory tracker using a python class. While not completing all of the bonus brief it allows full user input and for the inventory to be easily saved and reloaded at a later date.
