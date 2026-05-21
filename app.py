import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from langchain_core.tracers.langchain import wait_for_all_tracers, LangChainTracer

# Page config
st.set_page_config(page_title="De-gravitize App", page_icon="🚀", layout="wide")

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.markdown("Provide your API keys to get started.")
api_key = st.sidebar.text_input("Google Gemini API Key", type="password", help="Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)")

st.sidebar.markdown("---")
st.sidebar.markdown("### LangSmith Tracing (Optional)")
langsmith_api_key = st.sidebar.text_input("LangSmith API Key", type="password", help="Enable tracing by providing your LangSmith key.")

if langsmith_api_key:
    import os
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = "summarizer"



# Main Content
st.title("🚀 De-gravitize Content")
st.markdown("Strip away the fluff, weight, and filler from your text, leaving only the dense core of insight.")

text_input = st.text_area("Content to De-gravitize", height=300, placeholder="Paste your long-form text here...")

prompt_template_str = """
You are an elite, high-efficiency information processor. Your job is to strip away the fluff, weight, and filler from the provided text, leaving only the dense core of insight. 

De-gravitize the following content based on these exact rules:

1. **The Core Vector (TL;DR):** Give me a single, high-impact sentence explaining *exactly* what this text is about. No preamble like "This article discusses..." Go straight to the meat.
2. **The Frictionless Takeaways:** Provide 3 to 5 hyper-concise bullet points. Bold the first 2-4 words of each bullet point to make it instantly skimmable. 
3. **The Hidden Gravity:** In one sentence, what is the underlying motive, bias, or ultimate "so what?" behind this text?

Tone: Direct, objective, and sharp. 
Constraints: Keep it short enough to read in under 30 seconds. Do not repeat concepts.

Text to de-gravitize:
{text}
"""

if st.button("De-gravitize Content", type="primary"):
    if not api_key:
        st.error("Please provide your Google Gemini API Key in the sidebar.")
        st.stop()
        
    if not text_input.strip():
        st.warning("Please paste some text content to process.")
        st.stop()
        
    try:
        with st.spinner("De-gravitizing content..."):
            # Initialize LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite",
                google_api_key=api_key,
                temperature=0.3
            )
            
            # Setup Prompt Template
            prompt = PromptTemplate(
                template=prompt_template_str,
                input_variables=["text"]
            )
            # Create LangChain pipeline
            chain = prompt | llm | StrOutputParser()
            
            # Execute
            if langsmith_api_key:
                client = Client(api_url="https://api.smith.langchain.com", api_key=langsmith_api_key)
                tracer = LangChainTracer(project_name="summarizer", client=client)
                response = chain.invoke({"text": text_input}, config={"callbacks": [tracer]})
                wait_for_all_tracers()
            else:
                response = chain.invoke({"text": text_input})
            
            st.success("Successfully processed!")
            st.markdown("### Results")
            st.markdown(response)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your API key, network connection, or try again later.")
