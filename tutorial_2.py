"""
Title: tutorial_2.py
Date: 9/24/2019
Created by: Kristopher Ward
Revisions:

Purpose:
    To teach python classes
"""

def test(x):
    print(x)
    return x+1

class KrissClass:
    def __init__(self):
        print("Autumn uses atom")
        self.count = 0
    def increment(self):
        self.count += 1
    def print_count(self):
        print(self.count)

if __name__ == "__main__":
    # myObject = KrissClass()
    # for x in range(10):
    #     myObject.increment()
    #     myObject.print_count()
   variable = test(test(10))
   print(variable)

