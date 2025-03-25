from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from chatbot import dummy_get_session_history


# 1、构建知识索引库
# 1.1、从pdf文件中提取文本

def get_text_from_pdf(pdf_file_path: str) -> list:
    """从pdf文件中提取文本"""
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    return docs


# 1.2、将文档分割为小块文本
def split_text(docs: list, chunk_size: int = 500, chunk_overlap: int = 20) -> list:
    """将文本分割为小块文本"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)


# 1.3、创建向量数据库
def create_vectorstore(texts: list) -> object:
    """创建向量数据库"""
    # 创建OpenAIEmbeddings对象
    embedding = OpenAIEmbeddings()
    return Chroma.from_documents(texts, embedding)


# 2、创建对话链
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=1)
    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
    return chain


# 3、处理用户输入并生成回答
def handle_userinput(user_question):
    history = dummy_get_session_history(session_id='123')
    chain = get_conversation_chain(vectorstore)
    resp = chain({"question": user_question, "chat_history": history})
    print(resp['answer'])
    dummy_save_session_history(session_id='123', messages=resp['chat_history'])
    return resp['answer']
