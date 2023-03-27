from pathlib import Path

from langchain import LLMChain, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=4097-1024)

from llama_index import download_loader

UnstructuredReader = download_loader("UnstructuredReader")

loader = UnstructuredReader()
documents = loader.load_data(file=Path('PDX-2022-Year-end.pdf'))
documents = [d.to_langchain_format() for d in documents]
documents = text_splitter.split_documents(documents)

chat = ChatOpenAI()

# A prompt used to summarize a CV
map_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template="You are an investment analyst tasked with summarizing the Year-End report for your boss, who expects a concise summary with what has happened in the company. He doesn't want to see legal boilerplate",
                input_variables=[],
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="Please provide lists of the Opportunities, KPIs, General Events, and other relevant items. Make sure to capture the numbers. If there is nothing relevant in this section, you can say 'None'. \n\n {text} \n\n",
                input_variables=["text"],
            )
        ),
    ]
)

# A prompt used to determine the suitability of a CV
reduce_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template="You are an investment analyst tasked with summarizing the Year-End report for your boss, who expects a concise summary with what has happened in the company. He doesn't want to see legal boilerplate",
                input_variables=[],
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="Combine the summaries that your colleagues have provided into one combined document. \n\n {text} \n\n Make lists of  Opportunities, KPIs, General Events, and other relevant items. Make sure to capture the numbers.",
                input_variables=["text"],
            )
        ),
    ]
)

chain = load_summarize_chain(chat, chain_type="map_reduce", verbose=False, map_prompt=map_prompt, combine_prompt=reduce_prompt)
res = chain.run(documents)


# We now need to create some investment advice based on the summary
investment_advice_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template="You are an investment analyst tasked with summarizing the Year-End report for your boss, who expects a concise summary with what has happened in the company. He doesn't want to see legal boilerplate",
                input_variables=[],
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="Based on the summary, create recomendations for investors by applying the facts include \n\n {text} \n\n. How should investors react to this information?",
                input_variables=["text"],
            )
        )
    ]
)

chain = LLMChain(llm=chat, prompt=investment_advice_prompt)

reco = chain.run(text=res)

print("--- Summary ---")
print(res)
print("--- Recommendations ---")
print(reco)
