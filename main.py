import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedStyle
import algorithms
import pandas as pd

window = tk.Tk()
window.title("Project Preparation Workshop Assignment 3")

# Team names array
# region teams:
teamNames = [
    "KRC Genk",
    "Beerschot AC",
    "SV Zulte-Waregem",
    "Sporting Lokeren",
    "KSV Cercle Brugge",
    "RSC Anderlecht",
    "KAA Gent",
    "RAEC Mons",
    "FCV Dender EH",
    "Standard de Liège",
    "KV Mechelen",
    "Club Brugge KV",
    "KSV Roeselare",
    "KV Kortrijk",
    "Tubize",
    "Royal Excel Mouscron",
    "KVC Westerlo",
    "Sporting Charleroi",
    "Sint-Truidense VV",
    "Lierse SK",
    "KAS Eupen",
    "Oud-Heverlee Leuven",
    "Waasland-Beveren",
    "KV Oostende",
    "Manchester United",
    "Newcastle United",
    "Arsenal",
    "West Bromwich Albion",
    "Sunderland",
    "Liverpool",
    "West Ham United",
    "Wigan Athletic",
    "Aston Villa",
    "Manchester City",
    "Everton",
    "Blackburn Rovers",
    "Middlesbrough",
    "Tottenham Hotspur",
    "Bolton Wanderers",
    "Stoke City",
    "Hull City",
    "Fulham",
    "Chelsea",
    "Portsmouth",
    "Birmingham City",
    "Wolverhampton Wanderers",
    "Burnley",
    "Blackpool",
    "Swansea City",
    "Queens Park Rangers",
    "Norwich City",
    "Southampton",
    "Reading",
    "Crystal Palace",
    "Cardiff City",
    "Leicester City",
    "Bournemouth",
    "Watford",
    "AJ Auxerre",
    "FC Nantes",
    "Girondins de Bordeaux",
    "SM Caen",
    "Le Havre AC",
    "OGC Nice",
    "Le Mans FC",
    "FC Lorient",
    "Olympique Lyonnais",
    "Toulouse FC",
    "AS Monaco",
    "Paris Saint-Germain",
    "AS Nancy-Lorraine",
    "LOSC Lille",
    "Stade Rennais FC",
    "Olympique de Marseille",
    "FC Sochaux-Montbéliard",
    "Grenoble Foot 38",
    "Valenciennes FC",
    "AS Saint-Étienne",
    "RC Lens",
    "Montpellier Hérault SC",
    "US Boulogne Cote D'Opale",
    "AC Arles-Avignon",
    "Stade Brestois 29",
    "AC Ajaccio",
    "Évian Thonon Gaillard FC",
    "Dijon FCO",
    "Stade de Reims",
    "SC Bastia",
    "ES Troyes AC",
    "En Avant de Guingamp",
    "FC Metz",
    "Angers SCO",
    "GFC Ajaccio",
    "FC Bayern Munich",
    "Hamburger SV",
    "Bayer 04 Leverkusen",
    "Borussia Dortmund",
    "FC Schalke 04",
    "Hannover 96",
    "VfL Wolfsburg",
    "1. FC Köln",
    "Eintracht Frankfurt",
    "Hertha BSC Berlin",
    "DSC Arminia Bielefeld",
    "SV Werder Bremen",
    "FC Energie Cottbus",
    "TSG 1899 Hoffenheim",
    "Borussia Mönchengladbach",
    "VfB Stuttgart",
    "Karlsruher SC",
    "VfL Bochum",
    "SC Freiburg",
    "1. FC Nürnberg",
    "1. FSV Mainz 05",
    "1. FC Kaiserslautern",
    "FC St. Pauli",
    "FC Augsburg",
    "Fortuna Düsseldorf",
    "SpVgg Greuther Fürth",
    "Eintracht Braunschweig",
    "SC Paderborn 07",
    "FC Ingolstadt 04",
    "SV Darmstadt 98",
    "Atalanta",
    "Siena",
    "Cagliari",
    "Lazio",
    "Catania",
    "Genoa",
    "Chievo Verona",
    "Reggio Calabria",
    "Fiorentina",
    "Juventus",
    "Milan",
    "Bologna",
    "Roma",
    "Napoli",
    "Sampdoria",
    "Inter",
    "Torino",
    "Lecce",
    "Udinese",
    "Palermo",
    "Bari",
    "Livorno",
    "Parma",
    "Cesena",
    "Brescia",
    "Novara",
    "Pescara",
    "Hellas Verona",
    "Sassuolo",
    "Empoli",
    "Frosinone",
    "Carpi",
    "Vitesse",
    "FC Groningen",
    "Roda JC Kerkrade",
    "FC Twente",
    "Willem II",
    "Ajax",
    "N.E.C.",
    "De Graafschap",
    "FC Utrecht",
    "PSV",
    "Heracles Almelo",
    "Feyenoord",
    "Sparta Rotterdam",
    "ADO Den Haag",
    "FC Volendam",
    "SC Heerenveen",
    "AZ",
    "NAC Breda",
    "RKC Waalwijk",
    "VVV-Venlo",
    "Excelsior",
    "PEC Zwolle",
    "SC Cambuur",
    "Go Ahead Eagles",
    "FC Dordrecht",
    "Wisła Kraków",
    "Polonia Bytom",
    "Ruch Chorzów",
    "Legia Warszawa",
    "P. Warszawa",
    "Śląsk Wrocław",
    "Lechia Gdańsk",
    "Widzew Łódź",
    "Odra Wodzisław",
    "Lech Poznań",
    "GKS Bełchatów",
    "Arka Gdynia",
    "Jagiellonia Białystok",
    "Piast Gliwice",
    "Cracovia",
    "Korona Kielce",
    "Zagłębie Lubin",
    "Podbeskidzie Bielsko-Biała",
    "Pogoń Szczecin",
    "Zawisza Bydgoszcz",
    "Górnik Łęczna",
    "Termalica Bruk-Bet Nieciecza",
    "FC Porto",
    "CF Os Belenenses",
    "Sporting CP",
    "Trofense",
    "Vitória Guimarães",
    "Vitória Setúbal",
    "FC Paços de Ferreira",
    "SC Braga",
    "Amadora",
    "Académica de Coimbra",
    "Rio Ave FC",
    "SL Benfica",
    "Leixões SC",
    "CD Nacional",
    "Naval 1° de Maio",
    "CS Marítimo",
    "União de Leiria, SAD",
    "S.C. Olhanense",
    "Portimonense",
    "SC Beira Mar",
    "Feirense",
    "Gil Vicente FC",
    "Moreirense FC",
    "Estoril Praia",
    "FC Arouca",
    "FC Penafiel",
    "Boavista FC",
    "Uniao da Madeira",
    "Tondela",
    "Falkirk",
    "Rangers",
    "Heart of Midlothian",
    "Motherwell",
    "Kilmarnock",
    "Hibernian",
    "Aberdeen",
    "Inverness Caledonian Thistle",
    "Celtic",
    "St. Mirren",
    "Hamilton Academical FC",
    "Dundee United",
    "St. Johnstone FC",
    "Dunfermline Athletic",
    "Dundee FC",
    "Ross County FC",
    "Partick Thistle F.C.",
    "Valencia CF",
    "RCD Mallorca",
    "CA Osasuna",
    "Villarreal CF",
    "RC Deportivo de La Coruña",
    "Real Madrid CF",
    "CD Numancia",
    "FC Barcelona",
    "Racing Santander",
    "Sevilla FC",
    "Real Sporting de Gijón",
    "Getafe CF",
    "Real Betis Balompié",
    "RC Recreativo",
    "RCD Espanyol",
    "Real Valladolid",
    "Athletic Club de Bilbao",
    "UD Almería",
    "Atlético Madrid",
    "Málaga CF",
    "Xerez Club Deportivo",
    "Real Zaragoza",
    "CD Tenerife",
    "Hércules Club de Fútbol",
    "Levante UD",
    "Real Sociedad",
    "Granada CF",
    "Rayo Vallecano",
    "RC Celta de Vigo",
    "Elche CF",
    "SD Eibar",
    "Córdoba CF",
    "UD Las Palmas",
    "Grasshopper Club Zürich",
    "AC Bellinzona",
    "BSC Young Boys",
    "FC Basel",
    "FC Aarau",
    "FC Sion",
    "FC Luzern",
    "FC Vaduz",
    "Neuchâtel Xamax",
    "FC Zürich",
    "FC St. Gallen",
    "FC Thun",
    "Servette FC",
    "FC Lausanne-Sports",
    "Lugano"
]
teamNames.sort()

algorithmNames = ["RFC", "MLP", "DTC"]

# endregion
content_frame = ttk.Frame(window, padding=50, relief="flat", borderwidth=10)
content_frame.pack()

# Dropdown for First team name
team1_var = tk.StringVar(window)
team1_dropdown = ttk.Combobox(
    content_frame, textvariable=team1_var, values=teamNames)
team1_dropdown.pack(pady=10)

# Dropdown for Second team name
team2_var = tk.StringVar(window)
team2_dropdown = ttk.Combobox(
    content_frame, textvariable=team2_var, values=teamNames)
team2_dropdown.pack(pady=10)

# Algorithms
function_var = tk.StringVar(window)
function_dropdown = ttk.Combobox(
    content_frame, textvariable=function_var, values=["RFC", "MLP", "DTC"])
function_dropdown.pack(pady=10)


# gets data from UI
def getDataFromUI():
    team_1_name = team1_var.get()
    team_2_name = team2_var.get()
    algorithm = function_var.get()
    return team_1_name, team_2_name, algorithm


# validates UI data
def validateTeamsAndAlgorithm(team_1_name, team_2_name, algorithm):
    if team_1_name not in teamNames:
        raise Exception(f"'{team_1_name}' is not a valid team")
    if team_2_name not in teamNames:
        raise Exception(f"'{team_2_name}' is not a valid team")
    if algorithm not in algorithmNames:
        raise Exception(f"'{algorithm}' is not a valid algorithm")


# Function to calculate
def calculate(team1, team2, algorithm):
    try:
        validateTeamsAndAlgorithm(team1, team2, algorithm)
    except Exception as e:
        print(str(e))
        messagebox.showinfo("ERROR", f"{str(e)}")
        return

    if algorithm == "RFC":
        result = algorithms.prediction(team1, team2, "RFC")
    elif algorithm == "MLP":
        result = algorithms.prediction(team1, team2, "MLP")
    elif algorithm == "DTC":
        result = algorithms.prediction(team1, team2, "DTC")
    else:
        result = "No function selected"
    return result


# outputs calculate result to UI
def outputCalculateResult():
    team1, team2, algorithm = getDataFromUI()
    result = calculate(team1, team2, algorithm)

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

# Run the UI loop
window.mainloop()
