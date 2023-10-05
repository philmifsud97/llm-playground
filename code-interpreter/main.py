from json import tool
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import AgentType, create_csv_agent, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import PythonREPLTool,Tool
import langchain 
langchain.debug = True



def main():
    print("Strart")
    python_agent_executor = create_python_agent(llm= ChatOpenAI(temperature=0, model= "gpt-3.5-turbo"), tool= PythonREPLTool(),
                                                agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                                verbose= True,

                                                )

    # python_agent_executor.run("""generate and save in current working directory 15 QRCodes that point to 'www.youtube.com', you already have the qrcode python library installed
    #                           """)

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        path="code-interpreter/DataSources2.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    # csv_agent.run("How many input sources are in 'llm-playground/code-interpreter/DataSources2.csv'")

    grand_agent = initialize_agent(
        tools=[
            Tool(
                name="PythonAgent",
                func=python_agent_executor.run,
                description="""useful when you need to transform natural language and write from it python and execute the python code,
                              returning the results of the code execution,
                            DO NOT SEND PYTHON CODE TO THIS TOOL""",
            ),
            Tool(
                name="CSVAgent",
                func=csv_agent.run,
                description="""useful when you need to answer question over episode_info.csv file,
                             takes an input the entire question and returns the answer after running pandas calculations""",
            ),
        ],
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
    )

    try:
        response = grand_agent.run("""Generate and save in current working directory 15 QR codes that point to 'www.youtube.com', you have the qrcode package installed already """)
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        print(response)
    

if __name__ == "__main__":
    main()
