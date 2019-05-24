import sys
import os
import datetime as dt
import pandas as pd
import statsmodels.api as sm  # needed for Poisson regression model
import statsmodels.formula.api as smf  # needed for Poisson regression model
import numpy as np
from scipy.stats import poisson, skellam

# Main definition - constants
menu_actions = {}
season_year = "1819"

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

# ARGENTINA


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

    # print out all the teams
    print("Please pick home and away team:\n")
    for ref, team in argentina_teams.items():
        print(ref, ":", team)

    # input team
    homeChoice = input("Home Team Number: ")
    awayChoice = input("Away Team Number: ")

    # assign teams the object value within the dictionary
    homeTeam = argentina_teams[homeChoice]
    awayTeam = argentina_teams[awayChoice]

    goal_model_data = pd.concat([df_calculate[['Home', 'Away', 'HG']].assign(home=1).rename(
        columns={'Home': 'team', 'Away': 'opponent', 'HG': 'goals'}),
        df_calculate[['Away', 'Home', 'AG']].assign(home=0).rename(
        columns={'Away': 'team', 'Home': 'opponent', 'AG': 'goals'})])

    calculate_model(df_calculate, homeTeam, awayTeam, goal_model_data)

# AUSTRIA


def austria():
    austria_teams = {
        '1': 'AC Wolfsberger',
        '2': 'Admira',
        '3': 'Altach',
        '4': 'Austria Vienna',
        '5': 'Grodig',
        '6': 'Hartberg',
        '7': 'LASK Linz',
        '8': 'Mattersburg',
        '9': 'Neustadt',
        '10': 'Rapid Vienna',
        '11': 'Ried',
        '12': 'Salzburg',
        '13': 'St. Polten',
        '14': 'Sturm Graz',
        '15': 'Wacker Innsbruck'
    }

    df = pd.read_csv("http://www.football-data.co.uk/new/AUT.csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df_include = df[df['Date'].dt.year >= 2018]
    df_calculate = df_include[['Date', 'Home', 'Away', 'HG', 'AG', 'Res']]

    # print out all the teams
    print("Please pick home and away team:\n")
    for ref, team in austria_teams.items():
        print(ref, ":", team)

    # input team
    homeChoice = input("Home Team Number: ")
    awayChoice = input("Away Team Number: ")

    # assign teams the object value within the dictionary
    homeTeam = austria_teams[homeChoice]
    awayTeam = austria_teams[awayChoice]

    goal_model_data = pd.concat([df_calculate[['Home', 'Away', 'HG']].assign(home=1).rename(
        columns={'Home': 'team', 'Away': 'opponent', 'HG': 'goals'}),
        df_calculate[['Away', 'Home', 'AG']].assign(home=0).rename(
        columns={'Away': 'team', 'Home': 'opponent', 'AG': 'goals'})])

    calculate_model(df_calculate, homeTeam, awayTeam, goal_model_data)

# BELGIUM


def belgium():
    belgium_teams = {
        '1': 'Anderlecht',
        '2': 'Antwerp',
        '3': 'Cercle Brugge',
        '4': 'Charleroi',
        '5': 'Club Brugge',
        '6': 'Eupen',
        '7': 'Genk',
        '8': 'Gent',
        '9': 'Kortrijk',
        '10': 'Lokeren',
        '11': 'Mouscron',
        '12': 'Oostende',
        '13': 'St Truiden',
        '14': 'Standard',
        '15': 'Waasland-Beveren',
        '16': 'Waregem'
    }

    df = pd.read_csv(
        "http://www.football-data.co.uk/mmz4281/{}/B1.csv".format(season_year))

    df['Date'] = pd.to_datetime(df['Date'])
    df_include = df[df['Date'].dt.year >= 2018]
    df_calculate = df_include[['Date', 'HomeTeam',
                               'AwayTeam', 'FTHG', 'FTAG', 'FTR']]

    # print out all the teams
    print("Please pick home and away team:\n")
    for ref, team in belgium_teams.items():
        print(ref, ":", team)

    # input team
    homeChoice = input("Home Team Number: ")
    awayChoice = input("Away Team Number: ")

    # assign teams the object value within the dictionary
    homeTeam = belgium_teams[homeChoice]
    awayTeam = belgium_teams[awayChoice]

    goal_model_data = pd.concat([df_calculate[['HomeTeam', 'AwayTeam', 'FTHG']].assign(home=1).rename(
        columns={'HomeTeam': 'team', 'AwayTeam': 'opponent', 'FTHG': 'goals'}),
        df_calculate[['AwayTeam', 'HomeTeam', 'FTAG']].assign(home=0).rename(
        columns={'AwayTeam': 'team', 'HomeTeam': 'opponent', 'FTAG': 'goals'})])

    calculate_model(df_calculate, homeTeam, awayTeam, goal_model_data)

# BRAZIL


def brazil():
    brazil_teams = {
        '1': 'America MG',
        '2': 'Athletico-PR',
        '3': 'Atletico GO',
        '4': 'Atletico-MG',
        '5': 'Atletico-PR',
        '6': 'Avai',
        '7': 'Bahia',
        '8': 'Botafogo RJ',
        '9': 'Ceara',
        '10': 'Chapecoense-SC',
        '11': 'Corinthians',
        '12': 'Coritiba',
        '13': 'Criciuma',
        '14': 'Cruzeiro',
        '15': 'CSA',
        '16': 'Figueirense',
        '17': 'Flamengo RJ',
        '18': 'Fluminense',
        '19': 'Fortaleza',
        '20': 'Goias',
        '21': 'Gremio',
        '22': 'Internacional',
        '23': 'Joinville',
        '24': 'Nautico',
        '25': 'Palmeiras',
        '26': 'Parana',
        '27': 'Ponte Preta',
        '28': 'Portuguesa',
        '29': 'Santa Cruz',
        '30': 'Santos',
        '31': 'Sao Paulo',
        '32': 'Sport Recife',
        '33': 'Vasco',
        '34': 'Vitoria'
    }

    df = pd.read_csv("http://www.football-data.co.uk/new/BRA.csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df_include = df[df['Date'].dt.year >= 2018]
    df_calculate = df_include[['Date', 'Home', 'Away', 'HG', 'AG', 'Res']]

    # print out all the teams
    print("Please pick home and away team:\n")
    for ref, team in brazil_teams.items():
        print(ref, ":", team)

    # input team
    homeChoice = input("Home Team Number: ")
    awayChoice = input("Away Team Number: ")

    # assign teams the object value within the dictionary
    homeTeam = brazil_teams[homeChoice]
    awayTeam = brazil_teams[awayChoice]

    # assign teams the object value within the dictionary
    goal_model_data = pd.concat([df_calculate[['Home', 'Away', 'HG']].assign(home=1).rename(
        columns={'Home': 'team', 'Away': 'opponent', 'HG': 'goals'}),
        df_calculate[['Away', 'Home', 'AG']].assign(home=0).rename(
        columns={'Away': 'team', 'Home': 'opponent', 'AG': 'goals'})])

    calculate_model(df_calculate, homeTeam, awayTeam, goal_model_data)

# CHINA


def china():
    china_teams = {
        '1': 'Beijing Guoan',
        '2': 'Beijing Renhe',
        '3': 'Changchun Yatai',
        '4': 'Chongqing Lifan',
        '5': 'Dalian Yifang F.C.',
        '6': 'Guangzhou Evergrande',
        '7': 'Guangzhou R&F',
        '8': 'Guizhou Zhicheng',
        '9': 'Hangzhou Greentown',
        '10': 'Hebei',
        '11': 'Henan Jianye',
        '12': 'Jiangsu Suning',
        '13': 'Liaoning',
        '14': 'Shandong Luneng',
        '15': 'Shanghai Shenhua',
        '16': 'Shanghai Shenxin',
        '17': 'Shanghai SIPG',
        '18': 'Shenzhen',
        '19': 'Shijiazhuang',
        '20': 'Tianjin Quanjian',
        '21': 'Tianjin Teda',
        '22': 'Tianjin Tianhai',
        '23': 'Wuhan Zall',
        '24': 'Yanbian',
        '25': 'Zhejiang Yiteng'
    }

    df = pd.read_csv("http://www.football-data.co.uk/new/CHN.csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df_include = df[df['Date'].dt.year >= 2018]
    df_calculate = df_include[['Date', 'Home', 'Away', 'HG', 'AG', 'Res']]

    # print out all the teams
    print("Please pick home and away team:\n")
    for ref, team in china_teams.items():
        print(ref, ":", team)

    # input team
    homeChoice = input("Home Team Number: ")
    awayChoice = input("Away Team Number: ")

    # assign teams the object value within the dictionary
    homeTeam = china_teams[homeChoice]
    awayTeam = china_teams[awayChoice]

    # assign teams the object value within the dictionary
    goal_model_data = pd.concat([df_calculate[['Home', 'Away', 'HG']].assign(home=1).rename(
        columns={'Home': 'team', 'Away': 'opponent', 'HG': 'goals'}),
        df_calculate[['Away', 'Home', 'AG']].assign(home=0).rename(
        columns={'Away': 'team', 'Home': 'opponent', 'AG': 'goals'})])

    calculate_model(df_calculate, homeTeam, awayTeam, goal_model_data)

# DENMARK


def denmark():
    denmark_teams = {
        '1': 'Aalborg',
        '2': 'Aarhus',
        '3': 'Brondby',
        '4': 'Esbjerg',
        '5': 'FC Copenhagen',
        '6': 'Helsingor',
        '7': 'Hobro',
        '8': 'Horsens',
        '9': 'Lyngby',
        '10': 'Midtjylland',
        '11': 'Nordsjaelland',
        '12': 'Odense',
        '13': 'Randers FC',
        '14': 'Silkeborg',
        '15': 'Sonderjyske',
        '16': 'Vejle',
        '17': 'Vendsyssel',
        '18': 'Vestsjaelland',
        '19': 'Viborg'
    }

    df = pd.read_csv("http://www.football-data.co.uk/new/DNK.csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df_include = df[df['Date'].dt.year >= 2018]
    df_calculate = df_include[['Date', 'Home', 'Away', 'HG', 'AG', 'Res']]

    # print out all the teams
    print("Please pick home and away team:\n")
    for ref, team in denmark_teams.items():
        print(ref, ":", team)

    # input team
    homeChoice = input("Home Team Number: ")
    awayChoice = input("Away Team Number: ")

    # assign teams the object value within the dictionary
    homeTeam = denmark_teams[homeChoice]
    awayTeam = denmark_teams[awayChoice]

    # assign teams the object value within the dictionary
    goal_model_data = pd.concat([df_calculate[['Home', 'Away', 'HG']].assign(home=1).rename(
        columns={'Home': 'team', 'Away': 'opponent', 'HG': 'goals'}),
        df_calculate[['Away', 'Home', 'AG']].assign(home=0).rename(
        columns={'Away': 'team', 'Home': 'opponent', 'AG': 'goals'})])

    calculate_model(df_calculate, homeTeam, awayTeam, goal_model_data)

# ENGLAND


def england():
    leagues = {
        '1': 'Premier League',
        '2': 'Championship League',
        '3': 'League 1',
        '4': 'League 2',
        '5': 'Conference League'
    }

    # Choose which league to model
    print("Please pick which league:\n")
    for ref, league in leagues.items():
        print(ref, ":", league)

    leagueChoice = input("Enter League Number: ")

    # Choose which file to download and teams associated with league to reduce memory usage
    if leagueChoice == '1':
        df = pd.read_csv(
            "http://www.football-data.co.uk/mmz4281/{}/E0.csv".format(season_year), encoding='ISO-8859-1')

        england_teams = {
            '1': 'Arsenal',
            '2': 'Bournemouth',
            '3': 'Brighton',
            '4': 'Burnley',
            '5': 'Cardiff',
            '6': 'Chelsea',
            '7': 'Crystal Palace',
            '8': 'Everton',
            '9': 'Fulham',
            '10': 'Huddersfield',
            '11': 'Leicester',
            '12': 'Liverpool',
            '13': 'Man City',
            '14': 'Man United',
            '15': 'Newcastle',
            '16': 'Southampton',
            '17': 'Tottenham',
            '18': 'Watford',
            '19': 'West Ham',
            '20': 'Wolves'
        }
    elif leagueChoice == '2':
        df = pd.read_csv(
            "http://www.football-data.co.uk/mmz4281/{}/E1.csv".format(season_year), encoding='ISO-8859-1')

        england_teams = {
            '1': 'Aston Villa',
            '2': 'Birmingham',
            '3': 'Blackburn',
            '4': 'Bolton',
            '5': 'Brentford',
            '6': 'Bristol City',
            '7': 'Derby',
            '8': 'Hull',
            '9': 'Ipswich',
            '10': 'Leeds',
            '11': 'Middlesbrough',
            '12': 'Millwall',
            '13': 'Norwich',
            '14': 'Nott\'m Forest',
            '15': 'Preston',
            '16': 'QPR',
            '17': 'Reading',
            '18': 'Rotherham',
            '19': 'Sheffield United',
            '20': 'Sheffield Weds',
            '21': 'Stoke',
            '22': 'Swansea',
            '23': 'West Brom',
            '24': 'Wigan'
        }
    elif leagueChoice == '3':
        df = pd.read_csv(
            "http://www.football-data.co.uk/mmz4281/{}/E2.csv".format(season_year), encoding='ISO-8859-1')

        england_teams = {
            '1': 'Accrington',
            '2': 'AFC Wimbledon',
            '3': 'Barnsley',
            '4': 'Blackpool',
            '5': 'Bradford',
            '6': 'Bristol Rvs',
            '7': 'Burton',
            '8': 'Charlton',
            '9': 'Coventry',
            '10': 'Doncaster',
            '11': 'Fleetwood Town',
            '12': 'Gillingham',
            '13': 'Luton',
            '14': 'Oxford',
            '15': 'Peterboro',
            '16': 'Plymouth',
            '17': 'Portsmouth',
            '18': 'Rochdale',
            '19': 'Scunthorpe',
            '20': 'Shrewsbury',
            '21': 'Southend',
            '22': 'Sunderland',
            '23': 'Walsall',
            '24': 'Wycombe'
        }
    elif leagueChoice == '4':
        df = pd.read_csv(
            "http://www.football-data.co.uk/mmz4281/{}/E3.csv".format(season_year), encoding='ISO-8859-1')

        england_teams = {
            '1': 'Bury',
            '2': 'Cambridge',
            '3': 'Carlisle',
            '4': 'Cheltenham',
            '5': 'Colchester',
            '6': 'Crawley Town',
            '7': 'Crewe',
            '8': 'Exeter',
            '9': 'Forest Green',
            '10': 'Grimsby',
            '11': 'Lincoln',
            '12': 'Macclesfield',
            '13': 'Mansfield',
            '14': 'Milton Keynes Dons',
            '15': 'Morecambe',
            '16': 'Newport County',
            '17': 'Northampton',
            '18': 'Notts County',
            '19': 'Oldham',
            '20': 'Port Vale',
            '21': 'Stevenage',
            '22': 'Swindon',
            '23': 'Tranmere',
            '24': 'Yeovil'
        }
    else:
        df = pd.read_csv(
            "http://www.football-data.co.uk/mmz4281/{}/EC.csv".format(season_year), encoding='ISO-8859-1')

        england_teams = {
            '1': 'Aldershot',
            '2': 'Barnet',
            '3': 'Barrow',
            '4': 'Boreham Wood',
            '5': 'Braintree Town',
            '6': 'Bromley',
            '7': 'Chesterfield',
            '8': 'Dag and Red',
            '9': 'Dover Athletic',
            '10': 'Eastleigh',
            '11': 'Ebbsfleet',
            '12': 'Fylde',
            '13': 'Gateshead',
            '14': 'Halifax',
            '15': 'Harrogate',
            '16': 'Hartlepool',
            '17': 'Havant & Waterlooville',
            '18': 'Leyton Orient',
            '19': 'Maidenhead',
            '20': 'Maidstone',
            '21': 'Salford',
            '22': 'Solihull',
            '23': 'Sutton',
            '24': 'Wrexham'
        }

    # Modify headers
    df['Date'] = pd.to_datetime(df['Date'])
    df_include = df[df['Date'].dt.year >= 2018]
    df_calculate = df_include[['Date', 'HomeTeam',
                               'AwayTeam', 'FTHG', 'FTAG', 'FTR']]

    # print out all the teams
    print("Please pick home and away team:\n")
    for ref, team in england_teams.items():
        print(ref, ":", team)

    # input team
    homeChoice = input("Home Team Number: ")
    awayChoice = input("Away Team Number: ")

    # assign teams the object value within the dictionary
    homeTeam = england_teams[homeChoice]
    awayTeam = england_teams[awayChoice]

    goal_model_data = pd.concat([df_calculate[['HomeTeam', 'AwayTeam', 'FTHG']].assign(home=1).rename(
        columns={'HomeTeam': 'team', 'AwayTeam': 'opponent', 'FTHG': 'goals'}),
        df_calculate[['AwayTeam', 'HomeTeam', 'FTAG']].assign(home=0).rename(
        columns={'AwayTeam': 'team', 'HomeTeam': 'opponent', 'FTAG': 'goals'})])

    calculate_model(df_calculate, homeTeam, awayTeam, goal_model_data)

# FINLAND


def finland():
    finland_teams = {
        '1': 'HIFK',
        '2': 'HJK',
        '3': 'Honka',
        '4': 'Ilves',
        '5': 'Inter Turku',
        '6': 'KPV Kokkola',
        '7': 'KuPS',
        '8': 'Lahti',
        '9': 'Mariehamn',
        '10': 'PS Kemi',
        '11': 'Rovaniemi',
        '12': 'SJK',
        '13': 'TPS',
        '14': 'VPS'
    }

    df = pd.read_csv("http://www.football-data.co.uk/new/FIN.csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df_include = df[df['Date'].dt.year >= 2018]
    df_calculate = df_include[['Date', 'Home', 'Away', 'HG', 'AG', 'Res']]

    # print out all the teams
    print("Please pick home and away team:\n")
    for ref, team in finland_teams.items():
        print(ref, ":", team)

    # input team
    homeChoice = input("Home Team Number: ")
    awayChoice = input("Away Team Number: ")

    # assign teams the object value within the dictionary
    homeTeam = finland_teams[homeChoice]
    awayTeam = finland_teams[awayChoice]

    goal_model_data = pd.concat([df_calculate[['Home', 'Away', 'HG']].assign(home=1).rename(
        columns={'Home': 'team', 'Away': 'opponent', 'HG': 'goals'}),
        df_calculate[['Away', 'Home', 'AG']].assign(home=0).rename(
        columns={'Away': 'team', 'Home': 'opponent', 'AG': 'goals'})])

    calculate_model(df_calculate, homeTeam, awayTeam, goal_model_data)

# FRANCE


def france():
    leagues = {
        '1': 'Le Championnat',
        '2': 'Division 2'
    }

    # Choose which league to model
    print("Please pick which league:\n")
    for ref, league in leagues.items():
        print(ref, ":", league)

    leagueChoice = input("Enter League Number: ")

    # Choose which file to download and teams associated with league to reduce memory usage
    if leagueChoice == '1':
        df = pd.read_csv(
            "http://www.football-data.co.uk/mmz4281/{}/F1.csv".format(season_year), encoding='ISO-8859-1')

        france_teams = {
            '1': 'Amiens',
            '2': 'Angers',
            '3': 'Bordeaux',
            '4': 'Caen',
            '5': 'Dijon',
            '6': 'Guingamp',
            '7': 'Lille',
            '8': 'Lyon',
            '9': 'Marseille',
            '10': 'Monaco',
            '11': 'Montpellier',
            '12': 'Nantes',
            '13': 'Nice',
            '14': 'Nimes',
            '15': 'Paris SG',
            '16': 'Reims',
            '17': 'Rennes',
            '18': 'St Etienne',
            '19': 'Strasbourg',
            '20': 'Toulouse'
        }
    else:
        df = pd.read_csv(
            "http://www.football-data.co.uk/mmz4281/{}/F2.csv".format(season_year), encoding='ISO-8859-1')

        france_teams = {
            '1': 'Ajaccio',
            '2': 'Ajaccio GFCO',
            '3': 'Auxerre',
            '4': 'Beziers',
            '5': 'Brest',
            '6': 'Chateauroux',
            '7': 'Clermont',
            '8': 'Grenoble',
            '9': 'Le Havre',
            '10': 'Lens',
            '11': 'Lorient',
            '12': 'Metz',
            '13': 'Nancy',
            '14': 'Niort',
            '15': 'Orleans',
            '16': 'Paris FC',
            '17': 'Red Star',
            '18': 'Sochaux',
            '19': 'Troyes',
            '20': 'Valenciennes'
        }

    # Modify headers
    df['Date'] = pd.to_datetime(df['Date'])
    df_include = df[df['Date'].dt.year >= 2018]
    df_calculate = df_include[['Date', 'HomeTeam',
                               'AwayTeam', 'FTHG', 'FTAG', 'FTR']]

    # print out all the teams
    print("Please pick home and away team:\n")
    for ref, team in france_teams.items():
        print(ref, ":", team)

    # input team
    homeChoice = input("Home Team Number: ")
    awayChoice = input("Away Team Number: ")

    # assign teams the object value within the dictionary
    homeTeam = france_teams[homeChoice]
    awayTeam = france_teams[awayChoice]

    goal_model_data = pd.concat([df_calculate[['HomeTeam', 'AwayTeam', 'FTHG']].assign(home=1).rename(
        columns={'HomeTeam': 'team', 'AwayTeam': 'opponent', 'FTHG': 'goals'}),
        df_calculate[['AwayTeam', 'HomeTeam', 'FTAG']].assign(home=0).rename(
        columns={'AwayTeam': 'team', 'HomeTeam': 'opponent', 'FTAG': 'goals'})])

    calculate_model(df_calculate, homeTeam, awayTeam, goal_model_data)


# CALCULATION OF MODEL OUTPUTS


def calculate_model(df_calculate, homeTeam, awayTeam, goal_model_data):
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
    decimalUnder35 = oddsConverter(under35)

    # print(poisson_model.summary())
    print("\n\nHome team: {}".format(home_team))
    print("Away team: {}".format(away_team))
    # print(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
    print("\nImplied Probability {} Wins: {:.2%}".format(homeTeam, homeWin))
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
    print("\n\nFair Value of {} Wins: {} and odds at: {}".format(homeTeam,
                                                                 moneylineHome, decimalHome))
    print("Fair Value of Draw: {} and odds at: {}".format(
        moneylineDraw, decimalDraw))
    print("Fair Value of {} Wins: {} and odds at: {}".format(awayTeam,
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
    '2': austria,
    '3': belgium,
    '4': brazil,
    '5': china,
    '6': denmark,
    '7': england,
    '8': finland,
    '9': france,
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
