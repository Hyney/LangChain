import os

from dotenv import load_dotenv, find_dotenv
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.llms.tongyi import Tongyi
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv(find_dotenv('ali_openai.env'))


df = pd.read_excel('./deer.xlsx', skiprows=[0, 1])
df = df[df['专业'].notna()]

loader = DataFrameLoader(df, page_content_column="专业")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=0)
splits = text_splitter.split_documents(loader.load())

# vectorstore = Chroma.from_documents(documents=loader.load(), embedding=OpenAIEmbeddings())
# 调用阿里的通用文本向量
embedding = DashScopeEmbeddings(model='text-embedding-v1')

vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)

retriever = vectorstore.as_retriever()


template = """
请根据提供的学历和所学专业从职位表中筛选出所有符合的职位信息。
要求:
1、学历必须严格符合
2、专业大类必须严格符合
3、专业和学历必须对应
{question}

职位表:
{context}
"""

message_prompt = ChatPromptTemplate.from_template(template)

llm = Tongyi(temperature=0.9, model=os.getenv('MODEL'))

# RunnablePassthrough允许将用户的问题传递给prompt和model
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | message_prompt | llm

resp = rag_chain.invoke('本科学历 机械类专业')
print(resp)

