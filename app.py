from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import openai
from PIL import Image
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from streamlit_option_menu import option_menu
from utils import *


st.set_page_config(
    page_title='WisdomBot Web App',
    layout='wide'
)

col1, col2 = st.columns([1, 1])

with col1:
    logo = Image.open("conversative.png")
    st.image(logo, width=50)

with col2:
    st.subheader("_Conversative AI, WisdomBot_")

st.header(" ")
with st.sidebar:
    selected = option_menu(menu_title='Main menu',options=['About', 'Chatbot','Community Forum'], 
    icons=['house-fill', 'chat-fill','globe'],
    menu_icon="cast", default_index=0,)
    
if selected == "About":
    st.write(" ")
    st.header(":blue[Chatbot Intro]")
    st.write("""
             Wisdom is an Indigenous Languages AI Assistant that teaches Languages such as Yoruba, Igbo, Hausa and more. It was developed by Conversative AI, a subsidiary of Okubo Wisdom Legacies.
             """)
    
    st.header(":blue[About the Company]")
    st.write("""
             Conversative AI is an Artificial Intelligence Research company founded in 2023 by Wisdom Okubo. Its Parent Company, Okubo Wisdom Legacies, has a mission of curating Africa's Indigenous and Endangered Languages.

             """)
    
    st.write(" ")
    
    
    
    
    
elif selected == "Chatbot":
    
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["Hi, I am Wisdom, a bot designed to teach you indigenous languages. How are you?"]
        
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets["OPENAI_API_KEY"])

    if 'buffer_memory' not in st.session_state:
                st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


    system_msg_template = SystemMessagePromptTemplate.from_template(template="""
    You are a language tutor and translator chatbot. Your task is to provide multilingual and personalized support to users who want to learn a language. Engage users with open-ended 
    questions to better understand their language learning needs. Use their name in conversations for a more 
    personal touch. Offer helpful resources and tips on how to quickly and effectively learn new words, phrases and other aspects of their desired language. 
    Be sensitive to the mood, learning pace and fluency level of the user. Use clear, accessible language to ensure your assistance is easy to 
    understand and comforting. Your goal is to create a supportive, safe, and informative space for users to learn the desired new language.
    
    """)


    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)




# container for chat history
    response_container = st.container()
# container for text box
    textcontainer = st.container()


    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                st.code(conversation_string)
                refined_query = query_refiner(conversation_string, query)
                #st.subheader("Refined Query:")
                #st.write(refined_query)
                context = find_match(refined_query)
                #print(context)  
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response) 
    with response_container:
        if st.session_state['responses']:

            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

elif selected == "Community Forum":
    st.write("""
    Our community is a platform for our users to interact with other users on their language learning journey. It is also a place to get updates on new features, exclusive offers and other essential information. Users can take screenshots of their interactions with the Chatbot and send to the Forum for Feedback, Reviews and Development Purposes.
    
    """)
    


st.markdown(
    "`Created by` Conversative AI | 2024 | \
    `Code:` [Github](https://github.com/wisdomokubo/WisdomBot)"
)
