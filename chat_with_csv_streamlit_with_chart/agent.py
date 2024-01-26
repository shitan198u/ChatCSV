# Import necessary libraries.
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.llms import Ollama
# from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Define a function to create Pandas DataFrame agent from a CSV file.
def create_pd_agent(filename: str):
    # Initiate a connection to the LLM from Azure OpenAI Service via LangChain.
    llm = GoogleGenerativeAI(model="gemini-pro",
                            google_api_key="AIzaSyBx0n-jeinFp3xT5Mxm2T9yOQdM4RKjboM",
                            max_output_tokens=2048,
                            top_p=0.7,
                            top_k=30,
                            temperature=0
                            )
    # llm = Ollama(model="openchat")

    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(filename)

    # Create a Pandas DataFrame agent from the CSV file.
    return create_pandas_dataframe_agent(llm, df, verbose=False)

# Define a function to query the agent.
def query_pd_agent(agent, query):
    prompt = (
        """
        You must need to use matplotlib library if required to create a any chart.

        If the query requires creating a chart, please save the chart as "./chart_image/chart.png" and "Here is the chart:" when reply as follows:
        {"chart": "Here is the chart:"}

        If the query requires creating a table, reply as follows:
        {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
        
        If the query is just not asking for a chart, but requires a response, reply as follows:
        {"answer": "answer"}
        Example:
        {"answer": "The product with the highest sales is 'Minions'."}
        
        Lets think step by step.

        Here is the query: 
        """
        + query
    )

    # Run the agent with the prompt.
    response = agent.run(prompt)

    # Return the response in string format.
    return response.__str__()
