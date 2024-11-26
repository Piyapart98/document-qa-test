import streamlit as st

import pickle
import warnings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

warnings.filterwarnings("ignore", category=DeprecationWarning)

@st.cache_resource
def load_vectorstore():
    vector_file = 'vectorstore/vectorstore.pkl'
    with open(vector_file, "rb") as f:
        return pickle.load(f)

def get_source_from_doc(doc):
    pp = str(doc.metadata['page']+1)
    source = str(doc.metadata['source'])
    start = source.find('/')
    source = source[start+1:]
    text = source + ', page ' + pp
    return text
def format_docs(docs):
    text = ''
    for doc in docs:
        source = get_source_from_doc(doc)
        content = doc.page_content
        text += '\n\nSource: ' + source + '\n'+ content.strip() + '\n ----------'
    return text

# Load vectorstore object from a file and convert it into a retriever
vectorstore = load_vectorstore()
print("Vector store is loaded.")
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

template = '''
The following context contain source (PDF file name and page) and content inside, which are relevant technical details for answering the user's question. \n
CONTENT:\n
{context} \n
Answer the following question concisely and include all the relevant sources that you use at the end of the response in a form <filename, page> because these are the only sources available for the users. If the materials are not relevant or complete enough to confidently answer the user’s question, your best response is “the materials do not appear to be sufficient to provide a good answer.” \n
QUESTION: \n
{question}
'''
prompt = ChatPromptTemplate.from_template(template)

#  Streamlit app layout
with st.sidebar:
    "[![View the source code](https://img.shields.io/badge/Source%20Code-GitHub-blue?logo=github)](https://github.com/Piyapart98/document-qa-test/blob/main/streamlit_app.py)"

st.title("MARIA1.0")

"""
**M**eter **A**nalysis and **R**etrieval **I**nformation **A**ssistant: **MARIA** version 1.0

Developed by Piyapart Buttamart.
"""

if "messages" not in st.session_state:
    # Display in the first chat UI
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can retrieve and acquire data from PDFs. How can I assit you today?"}
    ]

for msg in st.session_state.messages:
    # Display a chat-like UI 
    st.chat_message(msg["role"]).write(msg["content"])
    # role = "user" or "assistant"

if query := st.chat_input(placeholder="Ask anything about meter"):
    # User input handling
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    llm = Ollama(
    model="llama3.2",
    temperature=0.7,   # Adjusts randomness
    top_k=40,          # Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
    top_p=0.5,         # Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
    verbose=False,
    cache=True
    )
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} # input: query, output: context (a formatted string of retrieved doc content) and question (the  query)
        | prompt # input: context and question, output: filled prompt
        | llm # input: filled prompt, output: string
        | StrOutputParser() # input: string, output: processed readable response
    )

    response_stream = rag_chain.stream(query)

    assistant_message = st.chat_message(name="assistant")

    # Stream the response directly using write_stream()
    response_text = assistant_message.write_stream(response_stream)
    
    # Append the full response to session state after streaming is complete to the UI
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    print("Response END")

# Example question: Explain the purposes of signal converter, signal processor, and microprocessor