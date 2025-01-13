import streamlit as st
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
#load_dotenv()

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results= 1, doc_content_chars_max= 250)
wiki = WikipediaQueryRun(api_wrapper= api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results= 1, doc_content_chars_max= 250)
arxiv = ArxivQueryRun(api_wrapper= api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name = "Search")

#side bar for settings
st.title("Search APP using Tools & Agents")

st.sidebar.title("GROQ_API KEY")
api_key = st.sidebar.text_input("Enter GROQ api key to proceed:", type = "password")

if api_key:

    if "messages" not in st.session_state:
        st.session_state['messages'] = [{"role":"assistant", "content":"Hi, I am a chatbot for search type, let me know how can I help \n Enter GROQ API key to Proceed"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])
    
    if prompt:=st.chat_input(placeholder="what is machine learning?"):
        st.session_state.messages.append({'role':"user", "content":prompt})
        st.chat_message("user").write(prompt)
    
    
    
    llm = ChatGroq(groq_api_key = api_key, model_name = "Gemma2-9b-It", streaming= True)
    
    tools = [search,arxiv,wiki]
    
    search_agent = initialize_agent(tools, llm, agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors = True)
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks= [st_cb])
        st.session_state.messages.append({'role':'assistant', "content": response})
        st.write(response)





