import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from datetime import datetime


def import_raw_data():
    """
    Imports data from the ODI-2025.csv file without modifying it, aside from column names.

    :return: The contents of ODI-2025.csv as a pandas dataframe.
    """
    usecols = ['Timestamp',
               'What programme are you in?',
               'Have you taken a course on machine learning?',
               'Have you taken a course on information retrieval?',
               'Have you taken a course on statistics?',
               'Have you taken a course on databases?',
               'What is your gender?',
               'I have used ChatGPT to help me with some of my study assignments ',
               'When is your birthday (date)?',
               'How many students do you estimate there are in the room?',
               'What is your stress level (0-100)?',
               'How many hours per week do you do sports (in whole hours)? ',
               'Give a random number',
               'Time you went to bed Yesterday',
               'What makes a good day for you (1)?',
               'What makes a good day for you (2)?']
    names = ['timestamp',
             'programme',
             'machine_learning',
             'information_retrieval',
             'statistics',
             'databases',
             'gender',
             'chatgpt',
             'birthday',
             'student_estimate',
             'stress',
             'sports',
             'random_number',
             'bedtime',
             'good_day_1',
             'good_day_2']

    df = pd.read_csv('files/ODI-2025.csv', usecols=usecols, sep=';')
    df.columns = names
    return df


def save_dataframe_to_file(df):
    """
    Saves a pandas dataframe to the file DMT_Data.csv with separator ;.

    :param df: Pandas dataframe
    """
    df.to_csv('files/DMT_Data.csv', index=False, sep=';')


def import_clean_data():
    return pd.read_csv('files/DMT_Data.csv', sep=';')


def list_unique_programmes(filename):
    data = import_clean_data()
    print(data['programme'].unique())


def column_to_string(df, colname):
    cleaned_column = df[colname].apply(lambda x: ' '.join(x.split()))
    return ' '.join(cleaned_column)


def column_wordcloud(filename, column):
    data = pd.read_csv('files/' + filename, sep=';')
    column_string = column_to_string(data, column)
    wordcloud = WordCloud(width=800,
                          height=400,
                          background_color='white',
                          colormap='viridis'
                          ).generate(column_string)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(f'figures/{column}_wordcloud.png', dpi=300, bbox_inches='tight')


def check_ai(programme):
    return bool(re.search(r'\b(ai|artificial|artifical|inteligence|intelligence)\b', programme, re.IGNORECASE))


def check_cs(programme):
    return bool(re.search(r'\b(cs|computer science|comp sci)\b', programme, re.IGNORECASE))


def check_cls(programme):
    return bool(re.search(r'\b(cls|computational science)\b', programme, re.IGNORECASE))


def check_econometrics(programme):
    return bool(re.search(r'\b(econometrics|eor)\b', programme, re.IGNORECASE))


def check_finance(programme):
    return bool(re.search(r'\b(finance|fintech)\b', programme, re.IGNORECASE))


def check_business(programme):
    return bool(re.search(r'\b(business|BA)\b', programme, re.IGNORECASE))


def check_bioinformatics(programme):
    return bool(re.search(r'bio', programme, re.IGNORECASE))


def clean_programmes(df):
    for idx, programme in df['programme'].items():
        if check_ai(programme):
            df.at[idx, 'programme'] = 'Artificial Intelligence'
        elif check_cs(programme):
            df.at[idx, 'programme'] = 'Computer Science'
        elif check_cls(programme):
            df.at[idx, 'programme'] = 'Computational Science'
        elif check_econometrics(programme):
            df.at[idx, 'programme'] = 'Econometrics'
        elif check_finance(programme):
            df.at[idx, 'programme'] = 'Finance'
        elif check_business(programme):
            df.at[idx, 'programme'] = 'Business Analytics'
        elif check_bioinformatics(programme):
            df.at[idx, 'programme'] = 'Bioinformatics'
        else:
            print(programme)
    return df


def convert_course_experience(df):
    for column in ['machine_learning', 'information_retrieval', 'statistics', 'databases']:
        for idx, value in df[column].items():
            if value in ['yes', '1', 'mu', 'ja']:
                df.at[idx, column] = 'yes'
            elif value in ['no', '0', 'sigma', 'nee']:
                df.at[idx, column] = 'no'
    return df


def convert_birthdays(df):
    for idx, value in df['birthday'].items():
        value = str(value)
        if re.match(r'^\d{2}-\d{2}-\d{4}$', value):
            # dd-mm-yyyy
            df.at[idx, 'birthday'] = value.replace('-', '/')
        elif re.match(r'^\d{2}-\d{2}-\d{2}$', value):
            # mm-dd-yy
            mm, dd, yy = value.split('-')
            century = '20' if int(yy) <= 25 else '19'
            formatted_value = f"{dd}/{mm}/{century}{yy}"
            df.at[idx, 'birthday'] = formatted_value
        elif re.match(r'^\d{8}$', value):
            # ddmmyyyy
            if float(value[4:]) > 1925:
                formatted_value = f"{value[:2]}/{value[2:4]}/{value[4:]}"
                df.at[idx, 'birthday'] = formatted_value
        elif re.match(r'^\d{2}\.\d{2}\.\d{4}$', value):
            # dd.mm.yyyy
            df.at[idx, 'birthday'] = value.replace('.', '/')
        elif re.match(r'^\d{4}\.\d{2}\.\d{2}$', value):
            # yyyy.mm.dd
            yyyy, mm, dd = value.split('.')
            formatted_value = f"{dd}/{mm}/{yyyy}"
            df.at[idx, 'birthday'] = formatted_value
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            # yyyy-mm-dd
            yyyy, mm, dd = value.split('-')
            formatted_value = f"{dd}/{mm}/{yyyy}"
            df.at[idx, 'birthday'] = formatted_value
        elif re.match(r'^\d{2} \d{2} \d{4}$', value):
            # dd mm yyyy
            df.at[idx, 'birthday'] = value.replace(' ', '/')
    return df


def remove_non_numeric(df, column):
    df[column] = df[column].apply(
        lambda x: re.sub(r'[^0-9.,\-+]', '', x) if x.count('E') > 1 else re.sub(r'[^0-9.,\-+E]', '', x)
    )
    return df


def clean_and_convert_stress(df):
    df['stress'] = df['stress'].astype(str)
    df['stress'] = df['stress'].apply(lambda x: re.sub(',', '.', x))
    df = remove_non_numeric(df, 'stress')
    df['stress'] = df['stress'].replace('', np.nan)
    df['stress'] = df['stress'].astype(float)
    return df


def clean_and_convert_sports(df):
    df['sports'] = df['sports'].astype(str)
    for idx, value in df['sports'].items():
        if value == 'zero':
            # the one case that had value 'zero'
            df.at[idx, 'sports'] = str(0)
        elif bool(re.search(r'-', value)):
            # range a-b
            lower, higher = value.split('-')
            df.at[idx, 'sports'] = lower
        elif bool(re.search(r'[.,]', value)):
            # decimal (question asked for whole hours)
            integral, _ = re.split(r'[.,]', value, maxsplit=1)
            df.at[idx, 'sports'] = integral
    df = remove_non_numeric(df, 'sports')
    df['sports'] = df['sports'].replace('', np.nan)
    df['sports'] = df['sports'].astype(float)
    return df

def clean_and_convert_random_number(df):
    df['random_number'] = df['random_number'].astype(str)
    # international notation '.' = decimal, ',' = exponent of 10^3
    df['random_number'] = df['random_number'].apply(
        lambda x: re.sub(r'\.', '', x) if x.count('.') > 1 else x
    )
    df['random_number'] = df['random_number'].apply(
        lambda x: re.sub(r',', '.', x) if x.count(',') == 1 else x
    )
    df = remove_non_numeric(df, 'random_number')
    df['random_number'] = df['random_number'].replace('', np.nan)
    df['random_number'] = df['random_number'].astype(float)
    return df


def convert_to_military_time(time_string):
    formats = [
        "%H:%M",  # hh:mm
        "%Hh%M", # hh'h'mm
        "%Hu%M", # hh'u'mm
        "%H.%M",  # hh.mm
        "%I.%M",  # h.mm
        "%H-%M",  # hh-mm
        "%H",  # hh
        "%I",  # h
        "%H%M",  # hhmm
        "%I%M",  # hmm
        "%I%p",  # hAM/PM
        "%H%p",  # hhAM/PM
        "%H:%M %p",  # hh:mm AM/PM
        "%I:%M %p",  # h:mm AM/PM
        "%H %p",  # hh AM/PM
        "%I %p",  # hh AM/PM
        "%H:%M%p",  # hh:mmAM/PM
        "%I:%M%p",  # h:mmAM/PM
        "%I.%M%p",  # h.mmAM/PM
    ]
    for format in formats:
        try:
            return datetime.strptime(time_string.strip().lower(), format).strftime("%H:%M")
        except ValueError:
            continue
    print(f"Could not parse time_string: {time_string}")
    return time_string


def convert_bedtime(df):
    for idx, value in df['bedtime'].items():
        df.at[idx, 'bedtime'] = convert_to_military_time(value)
    return df


def clean_and_save(df):
    df.drop(df.tail(1).index,inplace=True)
    df = clean_programmes(df)
    df = convert_course_experience(df)
    df = convert_birthdays(df)
    df = remove_non_numeric(df, 'student_estimate')
    df = clean_and_convert_stress(df)
    df = clean_and_convert_sports(df)
    df = clean_and_convert_random_number(df)
    df = convert_bedtime(df)
    save_dataframe_to_file(df)


clean_and_save(import_raw_data())
