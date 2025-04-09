import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('/content/ODI-2025.xlsx')

print(f"Aantal records: {df.shape[0]}")
print(f"Aantal attributen: {df.shape[1]}")
print("Kolomnamen:", df.columns.tolist())

# load the Excel file
file_path = 'ODI-2025.xlsx'
df = pd.read_excel(file_path, sheet_name='ODI 2025')

# clean column names
df.columns = df.columns.str.strip()

# distributions
categorical_cols = [
    "What programme are you in?",
    "Have you taken a course on machine learning?",
    "Have you taken a course on information retrieval?",
    "Have you taken a course on statistics?",
    "Have you taken a course on databases?",
    "What is your gender?",
    "I have used ChatGPT to help me with some of my study assignments",
    "Time you went to bed Yesterday"
]

for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    df[col].value_counts().head(10).plot(kind='bar')
    plt.title(f"Distribution of {col}")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


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


def list_unique_programmes(filename):
    data = pd.read_csv('files/' + filename, sep=';')
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
    plt.savefig('figures/' + column + '_wordcloud.png')
