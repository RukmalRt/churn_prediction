import pandas as pd
from sqlalchemy import create_engine

def get_sql():
    user = 'root'
    password = 'root'
    host = 'localhost'
    database = 'ibm_customer_churn'

    connection_link = f'mysql+mysqlconnector://{user}:{password}@{host}/{database}'

    engine = create_engine(connection_link)
    dataframe = pd.read_sql('SELECT * FROM ml_view', engine)
    return dataframe
