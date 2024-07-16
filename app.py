import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables from.env file
load_dotenv()

def get_pdf_text(pdf_doc):
    pdf_reader = PdfReader(pdf_doc)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, vectorstore):
    conversation_chain = get_conversation_chain(vectorstore)
    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']

    st.write("### Chat History:")
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            st.markdown(f"**User:** {message.content}")
        else:
            st.markdown(f"**Bot:** {message.content}")

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")

    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = None
    if "selected_pdf" not in st.session_state:
        st.session_state.selected_pdf = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                st.session_state.pdf_docs = pdf_docs

    if st.session_state.pdf_docs:
        st.subheader("Uploaded PDFs:")
        pdf_options = [pdf.name for pdf in st.session_state.pdf_docs]
        selected_pdf = st.selectbox("Select a PDF to chat with", pdf_options)
        st.session_state.selected_pdf = selected_pdf
        st.session_state.chat_history = []  # Clear chat history
        st.session_state.user_question = ""  # Clear input field

        if selected_pdf:
            for pdf in st.session_state.pdf_docs:
                if pdf.name == selected_pdf:
                    raw_text = get_pdf_text(pdf)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vectorstore = get_vectorstore(text_chunks)

    user_question = st.text_input("Ask a question about your document:", value=st.session_state.user_question)
    if st.button("Submit"):
        if user_question and st.session_state.vectorstore:
            handle_userinput(user_question, st.session_state.vectorstore)

if __name__ == '__main__':
    main()