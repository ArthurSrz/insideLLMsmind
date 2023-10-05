# Import necessary libraries
import databutton as db
import streamlit as st
from streamlit_lottie import st_lottie

# Import Langchain modules
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents.tools import Tool
from langchain import OpenAI
from langchain.agents import initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# Streamlit UI Callback
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain
from langchain.chains.llm_symbolic_math.base import LLMSymbolicMathChain
from langchain.memory import ConversationBufferMemory

import openai


# Import modules related to streaming response
import os
import time

OPENAI_API_KEY = OPENAI_API = db.secrets.get(name="OPENAI_API")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY



st.title("Inside LLMs mind")
st.lottie("https://lottie.host/5db2cb4f-6dbb-4a39-b2e2-d7eca7dc2989/SeQeMwtKCR.json")
st.markdown("Inside LLMs mind is a small app to demonstrate that LLMs are not essentially **black boxes** that gives out smart anwers out of the fog. They can be carefully crafted and designed to answer specific needs.")
st.markdown("In this app, two LLMs have been chained. One can retrieve info on Internet, the other perform math calculations. Depending on your prompt it will use one of them.")


# Initialize the OpenAI language model and search tool
llm = OpenAI(temperature=0)
llm_symbolic_math = LLMSymbolicMathChain.from_llm(llm, verbose=True)
search = DuckDuckGoSearchRun()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize chat history if it doesn't already exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set up the tool for responding to general questions
tools = [
    Tool(
        name="Calculator",
        func=llm_symbolic_math.run,
        description="useful for when you need to answer questions about math.",
    )
]

# Set up the tool for performing internet searches
search_tool = Tool(
    name="DuckDuckGo Search",
    func=search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. Be specific with your input or ask about something that is new and latest.",
)
tools.append(search_tool)

# Initialize the Zero-shot agent with the tools and language model
conversational_agent = initialize_agent(
    agent="conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=10,
    memory = st.session_state.memory
)

question = st.chat_input("Ask me anything")

# Display previous chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process the user's question and generate a response
if question:
    # Display the user's question in the chat message container
    with st.chat_message("user"):
        st.markdown(question)
    # Add the user's question to the chat history
    st.session_state.messages.append({"role": "user", "content": question})

    
    with st.chat_message("assistance"):
        st_callback = StreamlitCallbackHandler(st.container())
        message_placeholder = st.empty()
        full_response = ""
        assistance_response = conversational_agent.run(question, callbacks=[st_callback])
        # st.markdown(assistance_response)
        
        # Simulate a streaming response with a slight delay
        for chunk in assistance_response.split():
            full_response += chunk + " "
            time.sleep(0.05)

            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")

        # Display the full response
        message_placeholder.info(full_response)
    
    # Add the assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
