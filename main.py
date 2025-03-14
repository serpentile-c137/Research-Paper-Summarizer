import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatGooglePalm
from langchain.chains import LLMChain
from io import BytesIO
import PyPDF2

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-key-if-not-using-env')

genai.configure()
model = "gemini-2.0-flash-lite"
google_genai_model = ChatGoogleGenerativeAI(model=model, temperature=0.5)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to generate summary using Google GenAI via LangChain
def generate_summary(text):
    """Generate summary using Google GenAI via LangChain."""
    template = """
    Please summarize the following research paper and provide a technical summary of key findings, methodologies, and conclusions:

    {text}
    """
    prompt = PromptTemplate(input_variables=["text"], template=template)
    llm_chain = LLMChain(prompt=prompt, llm=google_genai_model)
    summary = llm_chain.run({"text": text})
    return summary

# Function to summarize multiple research papers
def summarize_multiple_papers(pdf_files):
    """Summarize multiple research papers uploaded as PDF files."""
    full_text = ""
    for pdf_file in pdf_files:
        paper_text = extract_text_from_pdf(pdf_file)
        full_text += paper_text + "\n\n"
    
    # Generate summary for the combined text of all papers
    summary = generate_summary(full_text)
    return summary

# Function to summarize an individual paper
def summarize(file_path, model):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    llm = ChatGoogleGenerativeAI(model=model, temperature=0.5)
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    summary = chain.invoke(docs)
    return summary['output_text']

# Custom prompt analysis function
def custom_prompt_summary(file_path, model, custom_prompt):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    vectorDB_path = 'faiss_store'
    prompt_template = custom_prompt + """
    Answer the following question based only on the provided context, do not use any external information. Always give a detailed answer in a language such that the answer can be used in summary of the paper.:

    <context>
    {text}
    </context>
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=['text'])
    llm = ChatGoogleGenerativeAI(model=model, temperature=0.5)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(vectorDB_path)
    vectorstore = FAISS.load_local(vectorDB_path, embeddings, allow_dangerous_deserialization=True)
    
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    result = chain.invoke({"question": prompt.template})
    
    return result['answer']

# Streamlit App
def main():
    st.set_page_config(page_title="Research Paper Summarizer", page_icon="📄")
    st.title("Research Paper Summarizer 📄")

    # Single file upload widget that allows multiple files
    uploaded_files = st.file_uploader("Upload your research papers (PDF format)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        st.write(f"Processing {len(uploaded_files)} papers...")
        
        # Summarize the uploaded papers
        summary = summarize_multiple_papers(uploaded_files)
        
        st.subheader("Summary of Research Papers:")
        st.write(summary)

        # Custom analysis option
        custom_prompt = st.text_area("Enter Custom Prompt", "Read the entire paper and give summary from each section in detail.")
        
        if st.button("Generate Custom Analysis"):
            with st.spinner("Processing your request..."):
                st.session_state.custom_analysis = ""
                for uploaded_file in uploaded_files:
                    papers_dir = "papers"
                    os.makedirs(papers_dir, exist_ok=True)
                    file_path = os.path.join(papers_dir, uploaded_file.name)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.session_state.custom_analysis += custom_prompt_summary(file_path, model, custom_prompt) + "\n"
        
        if hasattr(st.session_state, 'custom_analysis') and st.session_state.custom_analysis:
            st.subheader("Custom Analysis")
            st.write(st.session_state.custom_analysis)

if __name__ == "__main__":
    main()
