"""
Title: tutorial_3.py
Date: 9/24/2019
Created by: Kristopher Ward
Revisions:

Purpose:
    To teach basic tKinter
"""

# libraries
import tkinter as tk

class myGUI:
    def __init__(self):
        self.master = tk.Tk()

        self.button = tk.Button(self.master, text="button1", bg="blue", fg="white", command=self.func_button_1)
        self.button.pack()
        self.button_2 = tk.Button(self.master, text="button2", bg="red", command=self.func_button_2)
        self.button_2.pack()
    def func_button_1(self):
        print("I am button 1")
    def func_button_2(self):
        print("I am not button 1 but I'm button 2")

if __name__ == "__main__":
    myWindow = myGUI()
    myWindow.master.mainloop()
