import tkinter as tk
from tkinter import ttk, messagebox
from consts import teamNames, algorithmNames
import algorithms
from algorithms import get_teams_with_match_row, get_all_test_teams, get_all_dates_of_matches


# validates UI data
def validateTeamsAndAlgorithm(team_1_name, team_2_name, algorithm):
    if team_1_name not in teamNames:
        raise Exception(f"'{team_1_name}' is not a valid team")
    if team_2_name not in teamNames:
        raise Exception(f"'{team_2_name}' is not a valid team")
    if algorithm not in algorithmNames:
        raise Exception(f"'{algorithm}' is not a valid algorithm")


# Function to calculate
def calculate(team1, team2, match_date, algorithm):
    try:
        validateTeamsAndAlgorithm(team1, team2, algorithm)
    except Exception as e:
        print(str(e))
        messagebox.showinfo("ERROR", f"{str(e)}")
        return None

    if algorithm == "RFC":
        result = algorithms.prediction(team1, team2, "RFC")
    elif algorithm == "MLP":
        result = algorithms.prediction(team1, team2, "MLP")
    elif algorithm == "DTC":
        result = algorithms.prediction(team1, team2, "DTC")
    else:
        result = "No function selected"
    return result


def run_application():
    window = tk.Tk()
    window.title("Project Preparation Workshop Assignment 3")
    content_frame = ttk.Frame(
        window, padding=50, relief="flat", borderwidth=10)
    content_frame.pack()

    # Dropdown for First team name
    team1_var = tk.StringVar(window)
    team1_dropdown = ttk.Combobox(
        content_frame, textvariable=team1_var, state="readonly")
    team1_dropdown.pack(pady=10)

    # Dropdown for Second team name
    team2_var = tk.StringVar(window)
    team2_dropdown = ttk.Combobox(
        content_frame, textvariable=team2_var, state="readonly")
    team2_dropdown.pack(pady=10)

    # Dropdown for date of match
    date_var = tk.StringVar(window)
    date_dropdown = ttk.Combobox(
        content_frame, textvariable=date_var, state="readonly")
    date_dropdown.pack(pady=10)

    # Algorithms
    function_var = tk.StringVar(window)
    function_dropdown = ttk.Combobox(
        content_frame, textvariable=function_var, values=algorithmNames, state="readonly")
    function_dropdown.pack(pady=10)

    def update_dropdown_values():
        team_1_name = team1_var.get()
        team_2_name = team2_var.get()

        team1_dropdown['values'] = get_all_test_teams()

        if team_1_name == "":
            team2_values = get_all_test_teams()
        else:
            team2_values = get_teams_with_match_row(team_1_name)
        team2_dropdown['values'] = team2_values

        if team_1_name != "" and team_2_name != "":
            date_values = get_all_dates_of_matches(team_1_name, team_2_name)
        else:
            date_values = []
            date_var.set("")
        date_dropdown['values'] = date_values

        window.after(100, update_dropdown_values)

    # gets data from UI

    def getDataFromUI():
        team_1_name = team1_var.get()
        team_2_name = team2_var.get()
        match_date = date_var.get()
        algorithm = function_var.get()
        return team_1_name, team_2_name, match_date, algorithm

    # outputs calculate result to UI
    def outputCalculateResult():
        team1, team2, match_date, algorithm = getDataFromUI()
        result = calculate(team1, team2, match_date, algorithm)

        result_output = None
        if result == 1:
            result_output = f"{team1} wins"
        elif result == -1:
            result_output = f"{team2} wins"
        elif result == 0:
            result_output = 'draw'

        display_area.configure(state="normal")
        display_area.delete("1.0", tk.END)
        display_area.insert(tk.END, result_output)
        display_area.configure(state="disabled")

    # Button to trigger calculation
    calculate_button = ttk.Button(
        content_frame, text="Calculate", command=outputCalculateResult)
    calculate_button.pack(pady=10)

    # Display area
    display_area = tk.Text(window, height=5, width=30, state="disabled")
    display_area.pack(pady=10)

    update_dropdown_values()

    # Run the UI loop
    window.mainloop()


# Check if the file is executed directly (not imported)
if __name__ == '__main__':
    run_application()
