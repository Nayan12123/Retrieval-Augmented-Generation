import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from embedding import get_embedding_function
import requests
import json
from dotenv import load_dotenv
load_dotenv()


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
## INSTRUCTIONS
Given the CONTEXT and QUESTION, answer the QUESTION strictly with respect to the given CONTEXT. If you don't know the answer,
strictly return the following response: "Answer cannot be found".
\n\n
### CONTEXT
{context}
\n\n
Answer the QUESTION based on the above CONTEXT, strictly following the given INSTRUCTIONS:
\n
### QUESTION:\n
 {question}

Strictly Return only the ANSWER and not anything else from the CONTEXT or QUESTION.  
"""
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)
    
    
def get_response(prompt, context,question,temperature=0.1):
    llama3  = HuggingFaceEndpoint(repo_id=MODEL,temperature=temperature,repetition_penalty = 1.5)
    llm_chain = LLMChain(prompt=prompt, llm=llama3,verbose=True)
    response = llm_chain.run(context = context, question = question)
    return response


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_text = get_response(prompt_template,context_text,query_text)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = {}
    # response_text = response_text.split("###")[0]
    formatted_response['Response'] = response_text
    print(response_text)
    formatted_response['Sources'] = sources
    # formatted_response = beautify_json(formatted_response)
    print(formatted_response)
    return formatted_response


if __name__ == "__main__":
    main()