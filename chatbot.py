import configs
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOpenAI


store = {}
llm = ChatOpenAI(temperature=1)


def dummy_get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# trimmer = trim_messages(
#     token_counter=llm,
#     max_tokens=45,
#     start_on="human",
#     include_system=False
# )

chain = llm

chain_with_history = RunnableWithMessageHistory(chain, dummy_get_session_history)
config = {"configurable": {"session_id": "1"}}

resp = chain_with_history.invoke(
    [HumanMessage("中国的首都是哪座城市")],
    config=config,
)
print(resp.content)
resp = chain_with_history.invoke(
    [HumanMessage("它比较著名的高校有哪些")],
    config=config,
)
print(resp.content)

# 流式数据
for resp in chain_with_history.stream([HumanMessage('这个地方有哪些著名建筑?')], config=config):
    print(resp.content)
