from dotenv import find_dotenv, load_dotenv

from langchain_community.llms.tongyi import Tongyi
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv(find_dotenv('ali_openai.env'))

documents = [
    Document(page_content="狗是伟大的伴侣, 以其忠诚和友好而文明", metadata={'source': '哺乳动物宠物文档'}),
    Document(page_content="猫是独立的宠物, 通常喜欢自己的空间", metadata={'source': '哺乳动物宠物文档'}),
    Document(page_content="金鱼是初学者的流行宠物, 只需要相对简单的护理", metadata={'source': '鱼类宠物文档'}),
    Document(page_content="鹦鹉是聪明的鸟类, 能够模仿人类的语言", metadata={'source': '鸟类宠物文档'}),
    Document(page_content="兔子是社交动物, 需要足够的空间跳跃", metadata={'source': '哺乳动物宠物文档'})
]

embedding = DashScopeEmbeddings(model='text-embedding-v1')
vector_store = DocArrayInMemorySearch.from_documents(documents, embedding=embedding)
result = vector_store.similarity_search('小猫', k=2)
print(result)

# 相似度查询: 返回相似的分数, 分数越低相似度越高
result = vector_store.similarity_search_with_score('橘猫')
print(result)

retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)
resp = retriever.invoke('黑狗')
print(resp)


message = """
使用提供的上下文回答这个问题。
{question}

上下文:
{context}
"""

message_prompt = ChatPromptTemplate.from_messages([HumanMessage(content=message)])

llm = Tongyi(temperature=0.9)

# RunnablePassthrough允许将用户的问题传递给prompt和model
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | message_prompt | llm

rag_chain.invoke('请介绍一下猫?')

