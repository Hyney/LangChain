import openai

import configs


def get_completion(prompt, model='gpt-3.5-turbo'):
    client = openai.OpenAI()
    messages = [{'role': "user", 'content': prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


result = get_completion('What is 1+1?')
print(result)

customer_email = """
Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! And to make matters worse, 
the warranty don't cover the cost of cleaning up me kitchen. I need yer help right now,matey!
"""

style = """American English in a calm and respectful tone"""

prompt = f"""Translate the text that is delimited by triple backticks into a style that is {style}, 
text: {customer_email}"""

response = get_completion(prompt)

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

chat = ChatOpenAI(temperature=0.0)
template = """Translate the text that is delimited by triple backticks into a style that is {style}, text: {text}"""
prompt_template = ChatPromptTemplate.from_template(template)

print(prompt_template.messages[0].prompt)

customer_message = prompt_template.format_messages(style=style, text=customer_email)

customer_response = chat(customer_message)
print(customer_response.content)

service_reply = """
Hey there customer, the warranty does not cover cleaning expense for your kitchen because it's your fault that you misused
your blender by forgetting to put the lid on before starting the blender. Tough luck! See ya!
"""

service_style_pirate = """a polite tone that speaks in English Pirate"""

service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply
)
print(service_messages[0].content)
service_response = chat(service_messages)
print(service_response.content)

customer_review = """
This leaf blower is pretty amazing. It has four settings:\
candle blower, gentle breeze, windy city, abd tornado.\
It arrived in two days, just in time for my wife's anniversary present.\
I think my wife liked it so much she was speechless. Ao far I've been the only one using it, and I've been using it \
every other morning to clear the leaves
"""

review_template = """
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else?
delivery_days: How many days did it take for the product to 
price_value: Extract any sentences about the value or price,

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}

{format_instructions}
"""

from langchain.output_parsers import ResponseSchema, StructuredOutputParser

gift_schema = ResponseSchema(name='gift', description='gift')
delivery_days_schema = ResponseSchema(name='delivery_days', description='delivery_days')
price_value_schema = ResponseSchema(name='price_value', description='price_value')

response_schema = [gift_schema, delivery_days_schema, price_value_schema]
parser = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = parser.get_format_instructions()

prompt_template = ChatPromptTemplate.from_template(template=review_template)
messages = prompt_template.format_messages(
    text=customer_review,
    format_instructions=format_instructions
)
print(messages[0].content)
_response = chat(messages)
print(_response.content)
output = parser.parse(_response.content)
print(output)

# 创建对话链
from langchain.chains import ConversationChain
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, mermory=memory, verbose=True)

conversation.predict(input='Hi, my name is HeYang')
conversation.predict(input="What is 1+1?")
conversation.predict(input='What is my name?')

# chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(temperature=0.9)
prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)
chain = LLMChain(llm=llm, prompt=prompt)
product = "Queen Size Shhet Set"
chain.run(product)

from langchain.chains import SimpleSequentialChain

llm = ChatOpenAI(temperature=0.9)
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}"
)
chain_one = LLMChain(llm=llm, prompt=first_prompt)

second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following company: {company_name}"
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)
overall_simple_chain.run(product)

# 多个输入/输出
from langchain.chains import SequentialChain

llm = ChatOpenAI(temperature=0.9)
first_prompt = ChatPromptTemplate.from_template(
    'Translate the following review to english: \n\n{review}'
)
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key='english_review')

second_prompt = ChatPromptTemplate.from_template(
    'Can you summarize the following review in 1 sentence: \n\n{english_review}'
)
chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key='summary')

third_prompt = ChatPromptTemplate.from_template(
    'What language is the following review: \n\n{review}'
)
chain_third = LLMChain(llm=llm, prompt=third_prompt, output_key='language')

fourth_prompt = ChatPromptTemplate.from_template(
    "Write a following up response to the following "
    "summary in the specified language: \n\nSummary: {summary}\n\nLanguage: {language}"
)
chain_fourth = LLMChain(llm=llm, prompt=fourth_prompt, output_key='followup_message')

overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_third, chain_fourth],
    input_variables=['review'],
    output_variables=['english_review', 'summary', 'followup_message'],
    verbose=True
)

review = "test is test"
overall_chain(review)

# 路由链
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.
Here is a question:
{input}"""

math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.
Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""

computer_science_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 
Here is a question:
{input}"""

# 创建4种提示词模板
prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template
    },
    {
        "name": "history",
        "description": "Good for answering history question",
        "prompt_template": history_template
    },
    {
        "name": "computer_science",
        "description": "Good for answering computer science question",
        "prompt_template": computer_science_template
    }
]

from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0.9)

destination_chains = {}
for p_info in prompt_infos:
    name = p_info['name']
    prompt_template = p_info['prompt_template']
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]

destinations_str = '\n'.join(destinations)

default_prompt = ChatPromptTemplate.from_template("{input}")

default_chain = LLMChain(llm=llm, prompt=default_prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.
<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```
REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.
<< CANDIDATE PROMPTS >>
{destinations}
<< INPUT >>
{{input}}
<< OUTPUT (remember to include the ```json)>>"""
# 创建一个提示词模板，包含destination和input两个变量

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=['input'],
    output_parser=RouterOutputParser()
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

chain.run('test')
