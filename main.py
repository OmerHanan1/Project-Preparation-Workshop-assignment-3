import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle
import algorithms
import pandas as pd

# Create the UI window
window = tk.Tk()
window.title("Python UI")

# Apply themed style
style = ThemedStyle(window)
style.set_theme("arc")  # You can choose other themes as well

# Team names array
teamNames = ["Team 1", "Team 2", "Team 3"]

# Frame for content
content_frame = ttk.Frame(window, padding=20)
content_frame.pack()

# Dropdown for First team name
team1_var = tk.StringVar(window)
team1_dropdown = ttk.Combobox(content_frame, textvariable=team1_var, values=teamNames)
team1_dropdown.pack(pady=10)

# Dropdown for Second team name
team2_var = tk.StringVar(window)
team2_dropdown = ttk.Combobox(content_frame, textvariable=team2_var, values=teamNames)
team2_dropdown.pack(pady=10)

# Algorithms
function_var = tk.StringVar(window)
function_dropdown = ttk.Combobox(content_frame, textvariable=function_var, values=["functionA", "functionB", "functionC"])
function_dropdown.pack(pady=10)

# Function to calculate
def calculate():
    team1 = team1_var.get()
    team2 = team2_var.get()
    selected_function = function_var.get()

    if selected_function == "functionA":
        result = algorithms.functionA(team1, team2)
    elif selected_function == "functionB":
        result = algorithms.functionB(team1, team2)
    elif selected_function == "functionC":
        result = algorithms.functionC(team1, team2)
    else:
        result = "No function selected"

    display_area.configure(state="normal")
    display_area.delete("1.0", tk.END) 
    display_area.insert(tk.END, result)
    display_area.configure(state="disabled")

# Button to trigger calculation
calculate_button = ttk.Button(content_frame, text="Calculate", command=calculate)
calculate_button.pack(pady=10)

# Display area
display_area = tk.Text(window, height=5, width=30, state="disabled")
display_area.pack(pady=10)

# Run the UI loop
window.mainloop()
