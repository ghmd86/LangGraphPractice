
from langchain_mistralai import ChatMistralAI
from langchain_core import __version__ as core_version
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

    model = ChatMistralAI(model_name="mistral-small-2603", temperature=0.6)
    parser = StrOutputParser()
    chain = prompt_template | model | parser

    result = chain.invoke({"question": "What is langChain?"})
    print(result)


def batch_execute():
    """ Demo batch chain using LCEL and runnable """

    prompt_template = ChatPromptTemplate.from_template(
        "Translate to hindi: {text}")

    model = ChatMistralAI(model_name="mistral-small-2603", temperature=0.6)
    parser = StrOutputParser()
    chain = prompt_template | model | parser
    inputs = [{"text": "Hello how are you?"}, {"text": "Where is the bag?"}]
    results = chain.batch(inputs)
    for result in zip(inputs, results):
        print(f"{result[0]}: {result[1]}")


def excessive():
    prompt_template = ChatPromptTemplate.from_template(
        "Generate marketing tagline: {product} targeting {audience}")

    model = ChatMistralAI(model_name="mistral-small-2603", temperature=0.6)
    parser = StrOutputParser()
    chain = prompt_template | model | parser
    input = {"product": "Chips", "audience": "Kids"}
    result = chain.invoke(input)
    print(result)

if __name__ == "__main__":
    # main()
    # batch_execute()
    excessive()