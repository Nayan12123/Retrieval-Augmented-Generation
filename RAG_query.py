import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import  ConversationChain
from langchain.memory import ConversationSummaryMemory
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

ONLY_QUESTION_PROMPT = """
\n\n
Answer the given below question:
\n
### QUESTION:\n
 {question}

"""
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


    
    
def get_response(question,context = None,temperature=0.1):
    
    llama3  = HuggingFaceEndpoint(repo_id=MODEL,temperature=temperature,repetition_penalty = 1.5)
    if context is None:
        prompt = ChatPromptTemplate.from_template(ONLY_QUESTION_PROMPT)
        # memory = ConversationSummaryMemory(llm= llama3,
        #  return_messages=True)
        llm_chain = LLMChain(prompt=prompt, llm=llama3,verbose=True)
        response = llm_chain.run(question = question)
        
    else:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        # memory = ConversationSummaryMemory(llm=llama3,
        #  return_messages=True)
        llm_chain = LLMChain(prompt=prompt, llm=llama3,verbose=True)
        response = llm_chain.run(context = context, question = question)
    return response


def query_rag(query_text: str,get_context = 1):
    # Prepare the DB.
    if get_context:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search_with_score(query_text, k=3)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        response_text = get_response(query_text,context_text)
        sources = [doc.metadata.get("id", None) for doc, _score in results]
    else:
        response_text = get_response(query_text)
        sources = None
    formatted_response = {}
    formatted_response['Response'] = response_text
    print(response_text)
    formatted_response['Sources'] = sources
    print(formatted_response)
    return formatted_response


