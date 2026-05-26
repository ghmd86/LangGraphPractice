from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


def message_types():
    model = ChatOllama(model="phi4")


    messages = [
        SystemMessage("You are a halal financial adviser"),
        HumanMessage("Where should I not invest?"),
        AIMessage("Don't invest in non Halal businesses"),
        HumanMessage("Which businesses are considered halal?")
    ]

    response = model.invoke(messages)

    print(response.content)

if __name__ == "__main__":
    print("-"*50)
    print("Multiple message conversation")
    print("-"*50)

class MovieReview(BaseModel):
    title:str = Field(description = "Title of the movie")
    review: str = Field(description = "Review of the movie")
    rating: int = Field(description = "Rating of the movie")

llm = ChatOllama(model="phi4")

structured_model = llm.with_structured_output(MovieReview)

response = structured_model.invoke("The Terminator is a classic scifi movie about how skynet takes over robots and fights the human. The movie is a all time hit and has above 9 rating")

print(response)



