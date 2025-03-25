import configs

import langchain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)


def qa_vector():
    """基于空间向量检索"""
    index = VectorstoreIndexCreator(vecotrestore_cls=DocArrayInMemorySearch).from_loaders([loader])
    query = "Please list all your shirts with sun protection in a table in markdown and summarize each one."
    response = index.query(query)
    return response


def qa_doc():
    """基于docs检索"""
    docs = loader.load()
    embeddings = OpenAIEmbeddings()

    # 将文本转换为向量
    # embed = embeddings.embed_query('Hi my name is test')

    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    query = "Please suggest a shirt with sunblocking"
    docs = db.similarity_search(query)
    llm = ChatOpenAI(temperature=0.0)

    qdocs = ''.join(doc.page_content for doc in docs)
    resp = llm.call_as_llm(
        f"{qdocs} Question: Please list all your shirts with sun protection in a table in markdown and summarize each one"
    )
    return resp


def qa_stuff():
    """stuff检索器"""
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0.0)
    qa_stuffer = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True
    )
    query = "Please list all your shirts with sun protection in a table in markdown and summarize each one"

    response = qa_stuffer.run(query)
    return response


def qa_eval():
    """评估"""
    data = loader.load()
    index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])
    llm = ChatOpenAI(temperature=0.0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=index.vectorstore.as_retriever(),
        verbose=True,
        chain_type_kwargs={'document_separator': '<<<<>>>>>'}
    )

    examples = [
        {'query': 'Do the Cozy Comfort Pullover Set have side pockets', 'answer': 'yes'},
        {'query': 'What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?', 'answer': 'The DownTek collection'}
    ]

    from langchain.evaluation.qa import QAGenerateChain

    example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI())
    new_examples = example_gen_chain.apply_and_parse([{'doc': t} for t in data[:5]])
    examples += new_examples

    # 手动测试
    # langchain.debug = True
    # qa.run(examples[0]['query'])
    #
    # langchain.debug = False

    from langchain.evaluation.qa import QAEvalChain

    predictions = qa.apply(examples)
    eval_chain = QAEvalChain.from_llm(llm)
    graded_outputs = eval_chain.evaluate(examples, predictions)

    for i, eg in enumerate(examples):
        print(f"Example {i}:")
        print("Question: " + predictions[i]['query'])
        print("Real Answer: " + predictions[i]['answer'])
        print("Predicted Answer: " + predictions[i]['result'])
        print("Predicted Grade: " + graded_outputs[i]['text'])


def qa_agent():
    from langchain.agents import load_tools, initialize_agent, AgentType
    from langchain_experimental.agents.agent_toolkits import create_python_agent
    from langchain_experimental.tools.python.tool import PythonREPLTool
    from langchain_experimental.utilities.python import PythonREPL

    llm = ChatOpenAI(temperature=0.0)
    tools = load_tools(['llm-math', 'wikipedia'], llm=llm)

    agent = initialize_agent(
        tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )
    agent('What is the 25% of 300?')

    question = "Tom M. Mitchell is an American computer scientist \
    and the Founders University Professor at Carnegie Mellon University (CMU)\
    what book did he write?"

    result = agent(question)

    # 使用python代理工具
    agent = create_python_agent(
        llm,
        tool=PythonREPLTool(),
        verbose=True
    )

    customer_list = [
        ['Harrison', 'Chase'],
        ['Lang', 'Chain'],
        ['Dolly', 'Too'],
        ['Elle', 'Elem'],
        ['Geoff', 'Fusion'],
        ['Trance', 'Former'],
        ['Jen', 'Ayai']
    ]
    agent.run(
       f"Sort these customers by last name and then first name and print the output: {customer_list}"
    )

    # 通过debug查看运行细节
    langchain.debug = True
    agent.run(
       f"Sort these customers by last name and then first name and print the output: {customer_list}"
    )
    langchain.debug = False

    # 自定义代理工具
    from langchain.agents import initialize_agent, tool
    from datetime import date

    @tool
    def time(text: str) -> str:
        return str(date.today())

    agent = initialize_agent(
        tools + [time],
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )
    try:
        result = agent("what's the date today?")
    except:
        print('exception on external access')


if __name__ == '__main__':
    qa_vector()