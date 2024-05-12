import streamlit as st
import subprocess
from htmlTemplate import css, bot_template, user_template
import shutil
import os
from RAG_query import*

def CreateVectorDB():
    command = ["python", "DB_helper.py","--reset"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # with process.stdout, process.stderr:
    #     for line in iter(process.stdout.readline, b''):
    #         print("stdout:", line.strip())
    #     for line in iter(process.stderr.readline, b''):
    #         print("stderr:", line.strip())
    # process.wait()
    exit_code = process.returncode
    print("Exit code:", exit_code)
    return


def handle_userinput():
    new_list = st.session_state.chat_history[::-1]
    for i, message in enumerate(new_list):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents or ask me anything:")
    

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if 'num_pdfs' not in st.session_state:
            if len(pdf_docs)>0:
                st.session_state.num_pdfs = len(pdf_docs)
        ## copy pdf_docs to datafolder.
        if st.button("Process"):
            with st.spinner("Processing"):
                for uploaded_file in pdf_docs:
                    if uploaded_file is not None:
                        with open(os.path.join("./data", uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        CreateVectorDB()
                        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                        
                
         
    
    if 'num_pdfs' not in st.session_state:
        if user_question:
            out = query_rag(user_question,get_context=0)
            st.session_state.chat_history.append(out['Response'])
            st.session_state.chat_history.append(user_question)
            handle_userinput()  
            
            
        
        
    else:
        if user_question:
            out = query_rag(user_question)
            st.session_state.conversation = out['Response']
            st.session_state.chat_history.append(out['Response'])
            st.session_state.chat_history.append(user_question)
            handle_userinput()  
        


if __name__ == '__main__':
    main()