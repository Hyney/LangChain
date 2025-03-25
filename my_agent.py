import os

from dotenv import load_dotenv, find_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.chat_models import ChatTongyi


load_dotenv(find_dotenv('./ali_openai.env'))


llm = ChatTongyi(temperature=0.0, model=os.getenv('MODEL'))

tools = load_tools(['wikipedia', 'llm-math'], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

resp = agent.invoke("Who is the current leader of China? What is the population of China? What is the population of China divided by the leader of China?")

print(resp)
