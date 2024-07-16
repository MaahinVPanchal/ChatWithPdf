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
        chat_model = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo")

        # Initialize memory for the conversation
        memory = ConversationBufferMemory(memory_key="history")

        # Response accumulator
        response_accumulator = []

        # Iterate over each chunk of text
        for chunk in texts:
            # Define the prompt for this chunk
            prompt = (
                "Please analyze the following balance sheet data:\n\n"
                f"{chunk}\n\n"  # Insert actual balance sheet text chunk here
                "Your analysis should include the following sections with detailed explanations and calculations:\n\n"
                
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
                "     - Current Ratio = Current Assets / Current Liabilities\n"
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

            # Generate hypothetical insights using OpenAI's Chat model
            conversation_chain = ConversationChain(
                llm=chat_model,
                memory=memory,
                input_key='input',
                output_key='output'
            )

            response = conversation_chain.run(input=prompt)
            response_accumulator.append(response)

        # Combine all responses into a single output
        final_response = "\n\n".join(response_accumulator)
        return final_response

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
