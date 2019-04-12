---
layout: single
classes: wide
date: 2019-03-15
title: "Advent of code 2018 - Day 5"
excerpt: "A simple walkthrough of my solution"
header:
  overlay_image: "images/advent-doors.png"
  overlay_filter: 0.5
permalink: /advent-2018/day5/
sidebar:
  nav: advent-2018-sidebar
---


# Day 5 : Alchemic Reduction
The challenge for day 5 is based around a chemical reaction, reducing a large string of characters on certain conditions.

[Day 5 instructions](https://adventofcode.com/2018/day/5)

I will run through my solution step by step, my full script can be found at the [end of this post](#script)

## Part 1
**Fully react the initial chemical, if two letters adjacent to each other are the same character and different cases (lower and upper) then those two characters react and are removed. Repeat this process and find the length of the remaining chemical**


```python
with open('day5input.txt') as file:
    chemical = file.read().strip()

print(len(chemical))
chemical[:100]
```

    50000





    'hHsSmMHhhHwWlLojYCclLyJtPpTZzqdFfDYymMjJxXQOiIiSbBsGLROorMmlgvVkiIKRrGxXgZteETzUunNbBAaWwplrRoOgGLlJ'



The chemical is 50000 characters long, consisting of lower and uppercase characters. To react letters must be the same and of different case, Aa, aA etc. I make a list of lower and uppercase letters, combine those into two lists containing all valid permutations for reactions and then add these together into a full reaction list.


```python
import string

l = list(string.ascii_lowercase)  # lowercase letters
u = list(string.ascii_uppercase)  # uppercase letters

lu = list(zip(l,u))   # [(a,A), (b,B), .. (z,Z)]
ul = list(zip(u,l))   # [(A,a), (B,b), .. (Z,z)]

lowerU = [''.join(x) for x in lu]   # [aA, bB, .. zZ]
upperL = [''.join(x) for x in ul]   # [Aa, Bb. .. Zz] 

reactions = lowerU + upperL
```

To react the chemical, I search the chemical for each possible reaction in my reactions list, removing any instances of that reaction by replacing it with nothing. This process is repeated until the whole reaction list is checked against the chemical and no reactions were found, at that point the loop is broken and the length of the remaining chemical returned.


```python
def fully_react(chemical):
    letsgo = True
    while letsgo:    
        counter = 0
        for r in reactions:
            if r in chemical:
                chemical = chemical.replace(r,'')
                counter+=1
        if counter == 0:
            letsgo = False
    return len(chemical)

print('Part 1 answer : ', fully_react(chemical))
```

    Part 1 answer :  11814


## Part 2 : Improving the polymer
**Which letter when removed from the chemical creates the smallest polymer after fully reacting the remaining chemical.**

Similar to the fully react function, part two simply requires removing every instance of a letter (upper and lowercase) from the initial chemical and then measuring the length of the remainder. 

I store each of the results in a list to show the remainders more clearly.


```python
import pprint

answers = []

for lower, upper in zip(l, u):
    modified_chemical = chemical.replace(lower,'').replace(upper,'')
    remainder = fully_react(modified_chemical)
    answers.append(remainder)
    
pprint.pprint(sorted(zip(answers,l)))
```

    [(4282, 'g'),
     (11296, 'l'),
     (11296, 'p'),
     (11298, 'w'),
     (11300, 's'),
     (11314, 'k'),
     (11316, 't'),
     (11324, 'd'),
     (11328, 'm'),
     (11328, 'n'),
     (11328, 'o'),
     (11330, 'u'),
     (11332, 'c'),
     (11332, 'e'),
     (11334, 'a'),
     (11340, 'y'),
     (11342, 'v'),
     (11350, 'f'),
     (11362, 'b'),
     (11362, 'j'),
     (11366, 'r'),
     (11368, 'h'),
     (11368, 'q'),
     (11374, 'i'),
     (11376, 'x'),
     (11392, 'z')]


'g' is the letter which when removed leaves the shortest polymer, the answer as requested by the instructions was the length of that polymer, 4282.

<a id="script"></a>

## Full script


```python
import string

with open('day5input.txt') as file:
    chemical = file.read().strip('\n')
    
l = list(string.ascii_lowercase)
u = list(string.ascii_uppercase)

lu = list(zip(l,u))
ul = list(zip(u,l))
lowerU = [''.join(x) for x in lu]
upperL = [''.join(x) for x in ul]

reactions = lowerU + upperL

def fully_react(chemical):
    letsgo = True
    while letsgo:    
        counter = 0
        for r in reactions:
            if r in chemical:
                chemical = chemical.replace(r,'')
                counter+=1
        if counter == 0:
            letsgo = False
    return len(chemical)

print(fully_react(chemical))

"""
Part Two
"""
answers = []

for lower, upper in zip(l, u):
    modified_chemical = chemical.replace(lower,'').replace(upper,'')
    remainder = fully_react(modified_chemical)
    answers.append(remainder)
    
print(sorted(zip(answers,l)))
```

    11814
    [(4282, 'g'), (11296, 'l'), (11296, 'p'), (11298, 'w'), (11300, 's'), (11314, 'k'), (11316, 't'), (11324, 'd'), (11328, 'm'), (11328, 'n'), (11328, 'o'), (11330, 'u'), (11332, 'c'), (11332, 'e'), (11334, 'a'), (11340, 'y'), (11342, 'v'), (11350, 'f'), (11362, 'b'), (11362, 'j'), (11366, 'r'), (11368, 'h'), (11368, 'q'), (11374, 'i'), (11376, 'x'), (11392, 'z')]

