from langchain.chains.question_answering.map_reduce_prompt import messages
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from langchain_core import __version__ as core_version
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama

model = init_chat_model(
    model="phi4",
    model_provider="ollama"
)
    # ChatOllama(model="phi4", base_url="http://localhost:11434",
    # temperature=0.3))
from dotenv import load_dotenv
load_dotenv()

print(f"Version: {core_version}")

# def main():
#     llm = ChatMistralAI(model_name="mistral-small-2603",temperature=0.6)
#     response = llm.invoke("Where are the tallest humans average population located?")
#     print(f"Hello from pythonproject! {response.content}")


def main():
    """ Demo basic chain using LCEL and runnable """
    # Component 1: Define the prompt template using LCEL

    prompt_template = ChatPromptTemplate.from_template("You are a helpful assistant answer in single sentence: {question}")

    # model = ChatMistralAI(model_name="mistral-small-2603", temperature=0.6)
    parser = StrOutputParser()
    chain = prompt_template | model | parser

    result = chain.invoke({"question": "What is langChain?"})
    print(result)


def batch_execute():
    """ Demo batch chain using LCEL and runnable """

    prompt_template = ChatPromptTemplate.from_template(
        "Translate to hindi: {text}")

    # model = ChatMistralAI(model_name="mistral-small-2603", temperature=0.6)
    parser = StrOutputParser()
    chain = prompt_template | model | parser
    inputs = [{"text": "Hello how are you?"}, {"text": "Where is the bag?"}]
    results = chain.batch(inputs)
    for result in zip(inputs, results):
        print(f"{result[0]}: {result[1]}")


def excessive():
    prompt_template = ChatPromptTemplate.from_template(
        "Generate marketing tagline: {product} targeting {audience}")

    # model = ChatMistralAI(model_name="mistral-small-2603", temperature=0.6)
    parser = StrOutputParser()
    chain = prompt_template | model | parser
    input = {"product": "Chips", "audience": "Kids"}
    result = chain.invoke(input)
    print(result)
def ollamaAgent():
    prompt_template = ChatPromptTemplate.from_template(
        "Translate to urdu: {text}")



    parser = StrOutputParser()
    chain = prompt_template | model | parser
    inputs = [{"text": "Hello how are you?"}, {"text": "Where is the bag?"}]
    results = chain.batch(inputs)
    for result in zip(inputs, results):
        print(f"{result[0]}: {result[1]}")

def exercise_multi_model():
    models = {
        "m_phi4": init_chat_model(model="phi4", model_provider="ollama"),
        "m_qwen3": init_chat_model(model="qwen3:0.6b", model_provider="ollama")
    }
    prompt = "What are the cheapest international places for vacations?"
    print(f"prompt: {prompt}")
    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"{model_name}: {response}")

def message_object():
    model = ChatMistralAI(model_name="mistral-small-2603", temperature=0.6)
    # using message objects (more control over roles)
    messages = [
        SystemMessage(content="You are a doctor"),
        HumanMessage(content="What are the things to avoid when having high fever")
    ]
    response = model.invoke(messages)
    print(response.content)

if __name__ == "__main__":
    # main()
    # batch_execute()
    # excessive()
    # ollamaAgent()
    # exercise_multi_model()
    message_object()