from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import tempfile

embedding_model = OllamaEmbeddings(model="all-minilm")
KNOWLEDGE_BASE="""
### LangGraph overview
LangGraph, created by LangChain, is an open source AI agent framework designed to build, deploy and manage complex generative AI agent workflows. It provides a set of tools and libraries that enable users to create, run and optimize large language models (LLMs) in a scalable and efficient manner. At its core, LangGraph uses the power of graph-based architectures to model and manage the intricate relationships between various components of an AI agent workflow.

What does all this information mean? The following example can offer a clearer understanding of LangGraph: Think about these graph-based architectures as a powerful configurable map, a “Super-Map.” Users can envision the AI workflow as being “The Navigator” of this “Super-Map.” Finally, in this example, the user is “The Cartographer.” In this sense, the navigator charts out the optimal routes between points on the “Super-Map,” all of which are created by “The Cartographer.”

To recap, optimal routes within the graph-based architectures (“Super-Map”) are charted and explored by using the AI workflow (“The Navigator”). This analogy is a great place to start understanding LangGraph—and if you like maps then you are welcome for the bonus opportunity to see someone use the word cartographer.

### LangGraph workflow
LangGraph illuminates the processes within an AI workflow, allowing full transparency of the agent’s state. Within LangGraph, the “state” feature serves as a memory bank that records and tracks all the valuable information processed by the AI system. It’s similar to a digital notebook where the system captures and updates data as it moves through various stages of a workflow or graph analysis.

For example, if you were running agents to monitor the weather, this feature could track the number of times it snowed and make suggestions based on changing snowfall trends. This observability of how the system works to complete complex tasks is useful for beginners to understand more about state management. State management is helpful when it comes to debugging as it allows the application’s state to be centralized, thus often shortening the overall process.

This approach allows for more effective decision-making, improved scalability and enhanced overall performance. It also allows for more engagement with individuals who might be new to these processes or prefer a clearer picture of what is going on behind the scenes.

LangGraph is also built on several key technologies, including LangChain, a Python framework for building AI applications. LangChain includes a library for building and managing LLMs. LangGraph also uses the human-in-the-loop approach. By combining these technologies with a set of APIs and tools, LangGraph provides users with a versatile platform for developing AI solutions and workflows including chatbots, state graphs and other agent-based systems.

Delve deeper into the world of LangGraph by exploring its key features, benefits and use cases. By the end of this article, you will have the knowledge and resources to take the next steps with LangGraph.

### Graph architecture
## Stateful graphs: A concept where each node in the graph represents a step in the computation, essentially devising a state graph. This stateful approach allows the graph to retain information about the previous steps, enabling continuous and contextual processing of information as the computation unfolds. Users can manage all LangGraph’s stateful graphs with its APIs.

## Cyclical graph: A cyclical graph is any graph that contains at least one cycle and is essential for agent runtimes. This means that there exists a path that starts and ends at the same node, forming a loop within the graph. Complex workflows often involve cyclic dependencies, where the outcome of one step depends on previous steps in the loop.

## Nodes: In LangGraph, nodes represent individual components or agents within an AI workflow. Nodes can be thought of as “actors” that interact with each other in a specific way. For example, to add nodes for tool calling, one can use the ToolNode. Another example, the next node, refers to the node that will be executed following the current one.

## Edges: Edges are a function within Python that determines which node to execute next based on the current state. Edges can be conditional branches or fixed transitions.
"""

llm = init_chat_model(model="phi4", model_provider="ollama")

def create_know_base():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE, metadata={"source": "langGraph_knowledge_base.md"})

    chunks = splitter.split_documents([doc])
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store

def demo_basic_rag():
    vector_store = create_know_base()
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the following context:
        {context}
        
        Question: {question}
        Answer: 
        
        Make sure to answer in a concise manner. If you are not sure say "I don't know"
        """
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is langGraph?",
        "Who is Donald?",
        "What are the components for langGraph?"
    ]

    for question in questions:
        answer = rag_chain.invoke(question)
        print(f"question: {question}")
        print(f"Answer: {answer}")


def demo_rag_with_sources():

    vector_store = create_know_base()
    retreiver = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2})

    def format_output(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i+1}] {source} \n {doc.page_content}")
    prompt = ChatPromptTemplate.from_template("""
    Answer the questions from context:
    {context}
    question (include source): {question}
    
    Answer: 
    """)
    rag_pipeline = (
        {"context" : retreiver | format_output,
         "question" : RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_pipeline.invoke("Explain the workflow of langGraph?")

    print(answer)


if __name__ == "__main__":
    # demo_basic_rag()
    demo_rag_with_sources()