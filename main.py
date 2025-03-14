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

def summarize(file_path, model):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    llm = ChatGoogleGenerativeAI(model=model, temperature=0.5)
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    summary = chain.invoke(docs)
    return summary['output_text']

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

def main():
    load_dotenv()
    st.title("📄 Research Paper Summarizer & Custom Analysis")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file:
        papers_dir = "papers"
        os.makedirs(papers_dir, exist_ok=True)
        file_path = os.path.join(papers_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        model = "gemini-2.0-flash-lite"
        
        if "summary" not in st.session_state:
            st.session_state.summary = ""
        if "custom_analysis" not in st.session_state:
            st.session_state.custom_analysis = ""
        
        if st.button("Summarize Paper"):
            with st.spinner("Generating Summary..."):
                st.session_state.summary = summarize(file_path, model)
        
        if st.session_state.summary:
            st.subheader("Summary")
            st.write(st.session_state.summary)
        
        custom_prompt = st.text_area("Enter Custom Prompt", "Read the entire paper and give summary from each section in detail.")
        
        if st.button("Generate Custom Analysis"):
            with st.spinner("Processing your request..."):
                st.session_state.custom_analysis = custom_prompt_summary(file_path, model, custom_prompt)
        
        if st.session_state.custom_analysis:
            st.subheader("Custom Analysis")
            st.write(st.session_state.custom_analysis)
    
if __name__ == "__main__":
    main()
