from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

if __name__ == "__main__":
    load_dotenv()

    print("Hello LangChain")
    information = """
    Association football, commonly known as football, or soccer,[a] is a team sport played between two teams of 11 players each, 
    who primarily use their feet to propel a ball around a rectangular field called a pitch. The objective of the game is to score more goals 
    than the opposing team by moving the ball beyond the goal line into a rectangular-framed goal defended by the opposing team.
    Traditionally, the game has been played over two 45-minute halves, for a total match time of 90 minutes.
    With an estimated 250 million players active in over 200 countries and territories, it is the world's most popular sport.
    """

    summary_template = """
    given the information {information} about an activy I want you to create:
    1. A short summary
    2. How much people in the world like it
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    res = chain.invoke(input={"information": information})

    print(res)
