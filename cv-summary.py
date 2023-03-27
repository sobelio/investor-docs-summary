import sys

from langchain import LLMChain, PromptTemplate
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from unstructured.partition.auto import partition


def extract_content_from_cv(path):
    """Extracts the content from a CV and returns a string of the content"""
    extract = partition(filename=path)
    extract_str = "\n\n".join([str(i) for i in extract])
    return extract_str


chat = ChatOpenAI()

# A prompt used to summarize a CV
summary_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template="You are a helpful assistant that summarizes CVs for job applicants. You are currently working on a {role} role ",
                input_variables=["role"],
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="Please summarize this CV for me. \n\n {cv} \n\n",
                input_variables=["cv"],
            )
        ),
    ]
)
# A prompt used to determine the suitability of a CV
relevance_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template="You are a helpful assistant tasked with judging the suitablity of CVs for job applicants. You are currently working on a {role} role",
                input_variables=["role"],
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="Please evaluate the below profile and indicate their `Suitability` for this role on a scale of 1-5 and write down your reasoning. \n\n {summary} \n\n",
                input_variables=["summary"],
            )
        ),
    ]
)

# Combine the two prompts into a single chain
combined_chain = SequentialChain(
    chains=[
        LLMChain(llm=chat, prompt=summary_prompt, output_key="summary"),
        LLMChain(llm=chat, prompt=relevance_prompt),
    ],
    input_variables=["role", "cv"],
    verbose=False,
)


def determine_suitability(cv_text, role):
    """Determines the suitability of a CV for a role"""
    res = combined_chain.run(
        role=role,
        cv=cv_text,
    )
    return res


def determine_suitabilty_for_cv_file(file, role):
    """Determines the suitability of a CV for a role"""
    cv_text = extract_content_from_cv(file)
    return determine_suitability(cv_text, role)


if __name__ == "__main__":
    # Grab the path from the command line
    if len(sys.argv) < 2:
        print("Please provide a path to a CV")
        exit(1)
    path = sys.argv[1]
    # Determine the suitability of the CV
    for title in [
        "Data Scientist",
        "Data Engineer",
        "Data Analyst",
        "Software Engineer",
        "Gardener",
        "Cook at McDonalds",
    ]:
        print(f"===== {title} =====")
        res = determine_suitabilty_for_cv_file(path, title)
        print(res)
        print()
