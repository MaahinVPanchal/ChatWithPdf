import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load environment variables from .env file
load_dotenv()

# Function to extract text from PDF
def pdf_to_text(pdf_file):
    try:
        if not pdf_file.name.endswith('.pdf'):
            st.error("Invalid file type. Please upload a PDF file.")
            return None
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Function to call OpenAI API for hypothetical insights
def get_hypothetical_insight(api_key, text):
    try:
        # Initialize the text splitter
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # Create a FAISS index
        db = FAISS.from_texts(texts, embeddings)

        # Initialize the chat model
        chat_model = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo")

        # Initialize memory for the conversation
        memory = ConversationBufferMemory(memory_key="history")

        # Define the detailed prompt for hypothetical insights
        # Use the extracted text to create a specific prompt
        prompt = (
            f"Please analyze the following balance sheet data : \n\n{text}\n\n"
            "Your analysis should include the following:\n\n"
            "Comparative Overview:\n"
            "- A side-by-side comparison of the balance sheets for the years mentioned in the data.\n"
            "- Highlight major changes in assets, liabilities, and equity.\n\n"
            "Assets:\n"
            "- Compare current assets and non-current assets for both years.\n"
            "- Identify significant changes in cash, accounts receivable, inventory, and fixed assets.\n\n"
            "Liabilities:\n"
            "- Compare current liabilities and non-current liabilities.\n"
            "- Highlight changes in accounts payable, short-term debt, and long-term debt.\n\n"
            "Equity:\n"
            "- Analyze changes in shareholders' equity.\n"
            "- Discuss any new equity issued, retained earnings, and dividends paid.\n\n"
            "Financial Ratios:\n"
            "- Calculate and compare key financial ratios, including:\n"
            "  - Current ratio\n"
            "  - Quick ratio\n"
            "  - Debt to equity ratio\n"
            "  - Return on equity (ROE)\n\n"
            "Trends and Insights:\n"
            "- Identify trends or patterns in the financial data.\n"
            "- Provide insights into the company’s financial health and stability.\n\n"
            "Conclusion:\n"
            "- Summarize the key findings from the analysis.\n"
            "- Provide recommendations based on the balance sheet comparison."
        )

        # Generate hypothetical insights using OpenAI's Chat model
        conversation_chain = ConversationChain(
            llm=chat_model,
            memory=memory,
            input_key='input',
            output_key='output'
        )
        
        response = conversation_chain.run(input=prompt)
        return response

    except Exception as e:
        st.error(f"Error accessing OpenAI API: {e}")
        return None

def main():
    # Define the correct password from environment variables
    CORRECT_PASSWORD = os.getenv('APP_PASSWORD')

    # Create a login page
    st.title("Login")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button and password == CORRECT_PASSWORD:
        st.success("Login successful!")
        session_state = st.session_state
        session_state.logged_in = True
    elif login_button:
        st.error("Incorrect password. Please try again.")

    # Check if logged in before displaying main app content
    if getattr(st.session_state, 'logged_in', False):
        # Proceed to the main app content
        st.title("Balance Sheet Analyzer")

        # File uploader
        uploaded_file = st.file_uploader("Upload your balance sheet PDF", type=["pdf"])

        if uploaded_file is not None:
            try:
                # Extract text from PDF
                extracted_text = pdf_to_text(uploaded_file)
                if extracted_text:
                    # Display extracted text on the left side
                    st.write("### Extracted Text")
                    st.text_area("", extracted_text, height=300)

                    # Analyze button
                    if st.button("Analyze"):
                        # Get hypothetical insights
                        api_key = os.getenv('OPENAI_API_KEY')
                        observation = get_hypothetical_insight(api_key, extracted_text)
                        if observation:
                            st.write("### Analysis")
                            st.write(observation)

            except Exception as e:
                st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
