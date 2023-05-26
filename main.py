import tkinter as tk
from tkinter import ttk, messagebox, Text
import algorithms
from tkinter.font import Font
from algorithms import get_teams_with_match_row, get_all_test_teams, get_all_dates_of_matches, prediction

algorithmNames = ["RFC", "MLP", "DTC"]


def calculate(team1, team2, match_date, algorithm):
    try:
        if algorithm == 'RFC':
            y_predict, true_label = prediction(team1, team2, match_date, "RFC")
        elif algorithm == 'MLP':
            y_predict, true_label = prediction(team1, team2, match_date, "MLP")
        elif algorithm == 'DTC':
            y_predict, true_label = prediction(team1, team2, match_date, "DTC")
        else:
            print("main::calculate::Error")
    except Exception as e:
        messagebox.showinfo(f"Error: {e}")

    messagebox.showinfo("Prediction", "Predicted Result: " +
                        str(y_predict) + "\nTrue Result: " + str(true_label))

    return y_predict, true_label


def run_application():
    window = tk.Tk()
    window.title("Project Preparation Workshop Assignment 3")
    content_frame = ttk.Frame(
        window, padding=50, relief="flat", borderwidth=10)
    content_frame.pack()

    team1_label = ttk.Label(content_frame, text="Home Team:")
    team1_label.pack()

    # Dropdown for First team name
    team1_var = tk.StringVar(window)
    team1_dropdown = ttk.Combobox(
        content_frame, textvariable=team1_var, state="readonly")
    team1_dropdown.pack(pady=10)

    team1_label = ttk.Label(content_frame, text="Away Team:")
    team1_label.pack()

    # Dropdown for Second team name
    team2_var = tk.StringVar(window)
    team2_dropdown = ttk.Combobox(
        content_frame, textvariable=team2_var, state="readonly")
    team2_dropdown.pack(pady=10)

    team1_label = ttk.Label(content_frame, text="Match Date:")
    team1_label.pack()

    # Dropdown for date of match
    date_var = tk.StringVar(window)
    date_dropdown = ttk.Combobox(
        content_frame, textvariable=date_var, state="readonly")
    date_dropdown.pack(pady=10)

    team1_label = ttk.Label(content_frame, text="Algorithm:")
    team1_label.pack()

    # Algorithms
    function_var = tk.StringVar(window)
    function_dropdown = ttk.Combobox(
        content_frame, textvariable=function_var, values=algorithmNames, state="readonly")
    function_dropdown.pack(pady=10)

    # Updates values according to values inserted
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
        calculate(team1, team2, match_date, algorithm)

    # Button to trigger calculation
    calculate_button = ttk.Button(
        content_frame, text="Calculate", command=outputCalculateResult)
    calculate_button.pack(pady=10)

    update_dropdown_values()

    # Run the UI loop
    window.mainloop()


# Check if the file is executed directly (not imported)
if __name__ == '__main__':
    run_application()
