from langchain.agents import create_pandas_dataframe_agent
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain_google_genai import GoogleGenerativeAI
import pandas as pd

df = pd.read_csv('sales_data.csv')

pd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

pd_agent.run("Name of the city of first two sales?")


csv_agent = create_csv_agent(OpenAI(temperature=0), 'sales_data.csv', verbose=True)

