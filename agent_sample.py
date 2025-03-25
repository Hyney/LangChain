from dotenv import find_dotenv, load_dotenv
from langchain_community.chat_models import ChatOpenAI, ChatTongyi
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv(find_dotenv('ali_openai.env'))


llm = ChatTongyi(temperature=0.0, model='qwen-plus')
# tools = load_tools(['wikipedia', 'llm-math'], llm=llm)
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# result = agent.invoke("1+1等于几")
# print(result)

tools = load_tools(["dalle-image-generator"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

message = ("一只可爱的英国短毛小猫的特写, 圆脸, 大眼睛, 蓬松的金棕色皮毛, 腹部较浅. 小猫有小而圆的耳朵, 粉红色的鼻子和黑色的爪垫. 它躺在柔软的、有纹理的表面上,"
           "好奇地微微倾斜着头, 向上凝视着. 背景稍微模糊, 将焦点集中在小猫可爱的特征上, 整体氛围温暖舒适.")

output = agent.invoke(message)