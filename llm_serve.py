import configs

from fastapi import FastAPI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes


system_template = "{text}"
prompt_template = ChatPromptTemplate.from_messages(('human', system_template))


llm = ChatOpenAI(temperature=1)

parser = StrOutputParser()

chain = prompt_template | llm | parser


app = FastAPI(title='LangChain Server', version='1.0', description="LangChain服务")

add_routes(app, chain, path='/openai')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)
