import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain


# Load environment variables from .env file
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

def get_hypothetical_insight(api_key, text, prompt):
    try:
        # Initialize the text splitter
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=15000,  # Adjust chunk size to fit within model limits
            chunk_overlap=500,
            length_function=len
        )
        texts = text_splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # Create a FAISS index
        db = FAISS.from_texts(texts, embeddings)

        # Initialize the chat model
        chat_model = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o")

        # Initialize memory for the conversation
        memory = ConversationBufferMemory(memory_key="history")

        # Response accumulator
        response_accumulator = []

        # Iterate over each chunk of text
        for chunk in texts:
            # Define the prompt for this chunk
            full_prompt = (
                "Please analyze the following balance sheet data:\n\n"
                f"{chunk}\n\n"  # Insert actual balance sheet text chunk here
                "Your analysis should include the following sections with detailed explanations and calculations:\n\n"
                + prompt +
                "\n\n"
            )

            # Generate hypothetical insights using OpenAI's Chat model
            conversation_chain = ConversationChain(
                llm=chat_model,
                memory=memory,
                input_key='input',
                output_key='output'
            )

            response = conversation_chain.run(input=full_prompt)
            response_accumulator.append(response)

        # Combine all responses into a single output
        final_response = "\n\n".join(response_accumulator)
        return final_response

    except Exception as e:
        st.error(f"Error accessing OpenAI API: {e}")
        return None

def main():
    st.set_page_config(page_title="Balance Sheet Analyzer", layout="wide")

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        # Login page
        st.title("Login")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")

        if login_button and password == os.getenv('APP_PASSWORD'):
            st.session_state.logged_in = True
            st.experimental_rerun()
        elif login_button:
            st.error("Incorrect password. Please try again.")
    else:
        # Proceed to the main app content
        st.title("Balance Sheet Analyzer")

        # Sidebar link to "Chat with PDF"
        st.sidebar.title("Options")

        # File uploader
        uploaded_files = st.file_uploader("Upload your balance sheet PDFs", type=["pdf"], accept_multiple_files=True)
        selected_pdf = st.selectbox("Select a PDF to analyze", [file.name for file in uploaded_files])

        if uploaded_files:
            try:
                # Extract text from selected PDF
                extracted_text = get_pdf_text(next(file for file in uploaded_files if file.name == selected_pdf))
                if extracted_text:
                    # Define prompts
                    prompt_one = (
                        "1. Comparative Overview:\n"
                        "   - Provide a side-by-side comparison of the balance sheets for the years mentioned in the data.\n"
                        "   - Highlight major changes in assets, liabilities, and equity with percentage changes where applicable.\n\n"
                        
                        "2. Assets Analysis:\n"
                        "   - Compare current assets and non-current assets for both years.\n"
                        "   - Identify significant changes in cash, accounts receivable, inventory, and fixed assets.\n"
                        "   - Provide reasons for significant changes and their potential impact on the company.\n\n"
                        
                        "3. Liabilities Analysis:\n"
                        "   - Compare current liabilities and non-current liabilities for both years.\n"
                        "   - Highlight changes in accounts payable, short-term debt, and long-term debt.\n"
                        "   - Discuss the implications of these changes on the company’s financial health.\n\n"
                        
                        "4. Equity Analysis:\n"
                        "   - Analyze changes in shareholders' equity.\n"
                        "   - Discuss any new equity issued, retained earnings, and dividends paid.\n"
                        "   - Explain the impact of these changes on the company’s financial stability and investor confidence.\n\n"
                        
                        "5. Financial Ratios Calculation and Comparison:\n"
                        "   - Calculate and compare key financial ratios using the following formulas for both years:\n"
                        "     - CurrentRatio = Current Assets / Current Liabilities\n"
                        "     - Quick Ratio = (Current Assets - Inventory) / Current Liabilities\n"
                        "     - Debt to Equity Ratio = Total Liabilities / Shareholders' Equity\n"
                        "     - Return on Equity (ROE) = Net Income / Shareholders' Equity\n"
                        "     - Return on Assets (ROA) = Net Income / Total Assets\n"
                        "     - Working Capital = Current Assets - Current Liabilities\n"
                        "     - Equity Ratio = Total Equity / Total Assets\n"
                        "     - Asset Turnover Ratio = Net Sales / Average Total Assets\n"
                        "     - Inventory Turnover Ratio = Cost of Goods Sold / Average Inventory\n"
                        "     - Accounts Receivable Turnover Ratio = Net Credit Sales / Average Accounts Receivable\n"
                        "   - Provide detailed calculations for each ratio and explain what each ratio indicates about the company’s financial health.\n\n"
                        
                        "6. Trends and Insights:\n"
                        "   - Identify trends or patterns in the financial data over the years.\n"
                        "   - Discuss any emerging trends in the company’s assets, liabilities, and equity.\n"
                        "   - Provide insights into the company’s financial health and stability, and discuss potential risks and opportunities.\n\n"
                        
                        "7. Conclusion and Recommendations:\n"
                        "   - Summarize the key findings from the analysis.\n"
                        "   - Provide specific recommendations based on the balance sheet comparison, including potential actions the company could take to improve its financial position.\n"
                        "   - Offer strategic advice to address identified risks and leverage opportunities for growth.\n\n"
                    )

                    prompt_two = (
                        "8. Liquidity Management:\n"
                        "   - Discuss the company's ability to meet its short-term obligations based on the Current Ratio and Quick Ratio.\n"
                        "   - Provide recommendations to improve liquidity if necessary.\n\n"
                        
                        "9. Leverage Management:\n"
                        "   - Analyze the company's debt levels and capital structure using the Debt to Equity Ratio and Equity Ratio.\n"
                        "   - Suggest strategies to optimize leverage if needed.\n\n"
                        
                        "10. Profitability Management:\n"
                        "   - Evaluate the company's profitability and efficiency in generating returns for shareholders using ROE and ROA.\n"
                        "   - Offer advice on how to enhance profitability based on the analysis.\n\n"
                        
                        "11. Asset Management:\n"
                        "   - Assess how effectively the company is using its assets to generate sales with the Asset Turnover Ratio.\n"
                        "   - Provide insights into improving asset utilization.\n\n"
                        
                        "12. Receivables Management:\n"
                        "   - Discuss the company's effectiveness in collecting receivables based on the Accounts Receivable Turnover Ratio.\n"
                        "   - Recommend improvements to receivables management if applicable.\n\n"
                    )

                    # Variables to store results
                    observation_one = None
                    observation_two = None

                    # Button to analyze with both prompts
                    if st.button("Analyze"):
                        # Get hypothetical insights with PromptOne
                        api_key = os.getenv('OPENAI_API_KEY')
                        observation_one = get_hypothetical_insight(api_key, extracted_text, prompt_one)

                        # Get hypothetical insights with Prompt Two
                        observation_two = get_hypothetical_insight(api_key, extracted_text, prompt_two)

                    # Display results
                    if observation_one or observation_two:
                        st.write("### Analysis Results")
                        if observation_one:
                            st.write(observation_one)
                        if observation_two:
                            st.write(observation_two)

                    # Chat interface on the left sidebar
                    st.sidebar.title("Chat with Balance Sheet")
                    user_question = st.sidebar.text_input("Ask question")
                    if user_question:
                        # Get response from OpenAI's Chat model
                        vectorstore = get_vectorstore(get_text_chunks(extracted_text))
                        handle_userinput(user_question, vectorstore)

            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")

if __name__ == "__main__":
    main()