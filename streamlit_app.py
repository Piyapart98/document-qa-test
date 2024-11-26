import streamlit as st

import pickle
import warnings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

warnings.filterwarnings("ignore", category=DeprecationWarning)
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
def generate_response(query, rag_chain):
    response = ""
    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)
        response += chunk
    return response
# Load vectorstore object from a file and convert it into a retriever
vector_file = 'vectorstore/vectorstore.pkl'
with open(vector_file, "rb") as f:
    vectorstore = pickle.load(f)
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

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="langchain_search_api_key_openai", type="password"
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("🔎 LangChain - Chat with search")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

if "messages" not in st.session_state:
    # Display in the first chat UI
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    # Display a chat-like UI 
    st.chat_message(msg["role"]).write(msg["content"])
    # role = "user" or "assistant"

if query := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    # User input handling
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(query)

    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
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
    # Processing user input wit hthe agent
    with st.chat_message("assistant"):
        # st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        # response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        response = generate_response(query, rag_chain)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)