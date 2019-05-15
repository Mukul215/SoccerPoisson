import sys
import os
# TODO: Create a seperate file to incorporate the variables and objects
# import teams
import datetime as dt
import pandas as pd
import statsmodels.api as sm  # needed for Poisson regression model
import statsmodels.formula.api as smf  # needed for Poisson regression model
import numpy as np
from scipy.stats import poisson, skellam

# Main definition - constants
menu_actions = {}
# season_year = 1819

# data = pd.read_csv("http://www.football-data.co.uk/mmz4281/{}/E0.csv".format(
#     season_year))

# show first 4 for testing purposes and display success message
# print(data[:4])
# print('\nThe program has run successfully...\n')

# =======================
#     MENUS FUNCTIONS
# =======================

# Main menu


def main_menu():
    os.system('clear')

    print("Welcome,\n")
    print("Please choose the league you want to start:")
    print('1. Argentina: Primera Division')
    print('2. Austria: Bundesliga')
    print('3. Belgium: Jupiler League')
    print('4. Brazil: Serie A')
    print('5. China: Super League')
    print('6. Denmark: Superliga')
    print('7. England: Premiership & Divs 1,2,3 & Conference')
    print('8. Finland: Veikkausliiga')
    print('9. France: Le Championnat & Division 2')
    print('10. Germany: Bundesligas 1 & 2')
    print('11. Greece: Ethniki Katigoria')
    print('12. Ireland: Premier Division')
    print('13. Italy: Serie A & B')
    print('14. Japan: J-League')
    print('15. Mexico: Liga MX')
    print('16. Netherlands: KPN Eredivisie')
    print('17. Norway: Eliteserien')
    print('18. Poland: Ekstraklasa')
    print('19. Portugal: Liga I')
    print('20. Romania: Liga 1')
    print('21. Russia: Premier League')
    print('22. Scotland: Premiership & Divs 1,2 & 3')
    print('23. Spain: La Liga (Premera & Segunda)')
    print('24. Sweden: Allsvenskan')
    print('25. Switzerland: Super League')
    print('26. Turkey: Ligi 1')
    print('27. USA: MLS')
    print("\n0. Quit the application...")
    choice = input(" >>  ")
    exec_menu(choice)


# Execute menu


def exec_menu(choice):
    os.system('clear')
    ch = choice.lower()
    if ch == '':
        menu_actions['main_menu']()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print("Invalid selection, please try again.\n")
            menu_actions['main_menu']()

# Argentina Menu


def argentina():
    argentina_teams = {
        '1': 'Arsenal Sarandi',
        '2': 'Velez Sarsfield',
        '3': 'Racing Club',
        '4': 'Colon Santa FE',
        '5': 'Quilmes',
        '6': 'Newells Old Boys',
        '7': 'Godoy Cruz',
        '8': 'San Lorenzo',
        '9': 'River Plate',
        '10': 'Tigre',
        '11': 'San Martin S.J.',
        '12': 'Lanus',
        '13': 'Estudiantes L.P.',
        '14': 'All Boys',
        '15': 'Belgrano',
        '16': 'Argentinos Jrs',
        '17': 'Union de Santa Fe',
        '18': 'Boca Juniors',
        '19': 'Atl. Rafaela',
        '20': 'Independiente',
        '21': 'Gimnasia L.P.',
        '22': 'Rosario Central',
        '23': 'Olimpo Bahia Blanca',
        '24': 'Defensa y Justicia',
        '25': 'Banfield',
        '26': 'Crucero del Norte',
        '27': 'Sarmiento Junin',
        '28': 'Huracan',
        '29': 'Aldosivi',
        '30': 'Nueva Chicago',
        '31': 'Temperley',
        '32': 'Patronato',
        '33': 'Atl. Tucuman',
        '34': 'Talleres Cordoba',
        '35': 'Chacarita Juniors',
        '36': 'San Martin T.'

    }

    df = pd.read_csv("http://www.football-data.co.uk/new/ARG.csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df_include = df[df['Date'].dt.year >= 2018]
    df_calculate = df_include[['Date', 'Home', 'Away', 'HG', 'AG', 'Res']]

    homeTeam = 'Huracan'
    awayTeam = 'Union de Santa Fe'

    calculate_model(df_calculate, homeTeam, awayTeam)


# CALCULATION OF MODEL OUTPUTS


def calculate_model(df_calculate, homeTeam, awayTeam):
    goal_model_data = pd.concat([df_calculate[['Home', 'Away', 'HG']].assign(home=1).rename(
        columns={'Home': 'team', 'Away': 'opponent', 'HG': 'goals'}),
        df_calculate[['Away', 'Home', 'AG']].assign(home=0).rename(
        columns={'Away': 'team', 'Home': 'opponent', 'AG': 'goals'})])

    # calculate poisson model
    poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data,
                            family=sm.families.Poisson()).fit()
    home_team = poisson_model.predict(pd.DataFrame(data={'team': homeTeam, 'opponent': awayTeam,
                                                         'home': 1}, index=[1]))
    away_team = poisson_model.predict(pd.DataFrame(data={'team': awayTeam, 'opponent': homeTeam,
                                                         'home': 0}, index=[1]))

    # create a matrix of 5x5 (goals) and probabilities
    max_goals = 5
    team_pred = [[poisson.pmf(i, team_avg) for i in range(
        0, max_goals+1)] for team_avg in [home_team, away_team]]

    model_array = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))

    # calculate the probability that home team wins
    homeWin = model_array[1][0] + model_array[2][0] + model_array[2][1] + model_array[3][0] + model_array[3][1] + model_array[3][2] + \
        model_array[4][0] + model_array[4][1] + model_array[4][2] + model_array[4][3] + model_array[5][0] + model_array[5][1] + model_array[5][2] + \
        model_array[5][3] + model_array[5][4]

    # calculate the probability that there is a draw
    outcomeDraw = model_array[0][0] + model_array[1][1] + model_array[2][2] + \
        model_array[3][3] + model_array[4][4] + model_array[5][5]

    # calculate the probability that the away team wins
    awayWin = model_array[0][1] + model_array[0][2] + model_array[0][3] + model_array[0][4] + model_array[0][5] + model_array[1][2] + model_array[1][3] + \
        model_array[1][4] + model_array[1][5] + model_array[2][3] + model_array[2][4] + model_array[2][5] + model_array[3][4] + model_array[3][5] + \
        model_array[4][5]

    # exact goals probability
    exact2 = model_array[0][2] + model_array[1][1] + model_array[2][0]
    exact3 = model_array[0][4] + model_array[4][0] + \
        model_array[3][1] + model_array[1][3]

    # calculate the probability that the match outcome is over/under a certain amount
    under25 = model_array[0][0] + model_array[0][1] + model_array[1][0] + model_array[1][1] + \
        model_array[2][0] + model_array[0][2]
    over25 = 1 - under25
    under35 = under25 + \
        model_array[0][3] + model_array[3][0] + \
        model_array[2][1] + model_array[1][2]
    over35 = 1 - under35

    # Convert all probabilities into moneyline and odds
    moneylineExact2 = moneylineConverter(exact2)
    decimalExact2 = oddsConverter(exact2)
    moneylineDraw = moneylineConverter(outcomeDraw)
    decimalDraw = oddsConverter(outcomeDraw)
    moneylineHome = moneylineConverter(homeWin)
    decimalHome = oddsConverter(homeWin)
    moneylineAway = moneylineConverter(awayWin)
    decimalAway = oddsConverter(awayWin)
    moneylineUnder25 = moneylineConverter(under25)
    decimalUnder25 = oddsConverter(under25)
    moneylineOver25 = moneylineConverter(over25)
    decimalOver25 = oddsConverter(over25)
    moneylineExact3 = moneylineConverter(exact3)
    decimalExact3 = oddsConverter(exact3)
    moneylineOver35 = moneylineConverter(over35)
    decimalOver35 = oddsConverter(over35)
    moneylineUnder35 = moneylineConverter(under35)
    decimalUnder35 = oddsConverter(over35)

    print(poisson_model.summary())
    print("Home team: {}".format(home_team))
    print("Away team: {}\n\n".format(away_team))
    print(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
    print("\n\n\nImplied Probability {} Wins: {:.2%}".format(homeTeam, homeWin))
    print("Implied Probability of Draw: {:.2%}".format(outcomeDraw))
    print("Implied Probability {} Wins: {:.2%}".format(awayTeam, awayWin))
    print(
        "Implied Probability Game Total is Under 2.5: {:.2%}".format(under25))
    print(
        "Implied Probability Game Total is Exactly 2: {:.2%}".format(exact2))
    print("Implied Probability Game Total is Over 2.5: {:.2%}".format(over25))
    print(
        "Implied Probability Game Total is Under 3.5: {:.2%}".format(under35))
    print("Implied Probability Game Total is Exactly 3: {:.2%}".format(exact3))
    print("Implied Probability Game Total is Over 3.5: {:.2%}".format(over35))
    print("\n\nFair Value of Home Team Wins: {} and odds at: {}".format(
        moneylineHome, decimalHome))
    print("Fair Value of Draw: {} and odds at: {}".format(
        moneylineDraw, decimalDraw))
    print("Fair Value of Away Team Wins: {} and odds at: {}".format(
        moneylineAway, decimalAway))
    print("Fair Value of Game Total Under 2.5: {} and odds at: {}".format(
        moneylineUnder25, decimalUnder25))
    print("Fair Value of Game Total Exactly 2: {} and odds at: {}".format(
        moneylineExact2, decimalExact2))
    print("Fair Value of Game Total Over 2.5: {} and odds at: {}".format(
        moneylineOver25, decimalOver25))
    print("Fair Value of Game Total Under 3.5: {} and odds at: {}".format(
        moneylineUnder35, decimalUnder35))
    print("Fair Value of Game Total Exactly 3: {} and odds at: {}".format(
        moneylineExact3, decimalExact3))
    print("Fair Value of Game Total Over 3.5: {} and odds at: {}".format(
        moneylineOver35, decimalOver35))
    print("\n\nPlease check betfair exchanges for the most accurate lines\n\n\n")

# Convert moneyline and adds from imlied value to fair value


def oddsConverter(probability):
    return round(1/probability, 2)


def moneylineConverter(probability):
    if probability >= .5:
        return round(-((probability/(1-probability))*100), 0)
    else:
        return round((((1-probability)/probability)*100), 0)

# Exit program


def exit():
    sys.exit()


# =======================
#    MENUS DEFINITIONS
# =======================

# Menu definition
menu_actions = {
    'main_menu': main_menu,
    '1': argentina,
    # '2': austria,
    # '3': belgium,
    # '4': brazil,
    # '5': china,
    # '6': denmark,
    # '7': england,
    # '8': finland,
    # '9': france,
    # '10': germany,
    # '11': greece,
    # '12': ireland,
    # '13': italy,
    # '14': japan,
    # '15': mexico,
    # '16': netherlands,
    # '17': norway,
    # '18': poland,
    # '19': portugal,
    # '20': romania,
    # '21': russia,
    # '22': scotland,
    # '23': spain,
    # '24': sweden,
    # '25': switzerland,
    # '26': turkey,
    # '27': usa,
    '0': exit
}

# =======================
#      MAIN PROGRAM
# =======================

# Main Program
if __name__ == "__main__":
    # Launch main menu
    main_menu()
