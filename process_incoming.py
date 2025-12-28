import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import json
import re
import math
import os

# --- Configuration & Setup ---

# Set Streamlit to wide layout for the minimalist chat feel
st.set_page_config(
    page_title="AIKARA: Unifying Knowledge",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load the embeddings data once per session
@st.cache_resource
def load_rag_assets():
    """Loads the pre-processed embeddings data."""
    try:
        # NOTE: Ensure 'embeddings.joblib' is in the same directory as this script.
        return joblib.load('embeddings.joblib')
    except FileNotFoundError:
        st.error("Error: 'embeddings.joblib' file not found. Please run your preprocessing script.")
        return None

df = load_rag_assets()

# --- Utility Functions ---

def format_seconds_to_mm_ss(seconds):
    """Converts a time in total seconds (float/str) to MM:SS format."""
    try:
        total_seconds = math.floor(float(seconds))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return "N/A" # Return N/A if conversion fails

def create_embedding(text_list):
    """Generates embeddings via the local Ollama API."""
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        }, timeout=10)
        r.raise_for_status()
        return r.json()["embeddings"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Ollama embedding service: {e}")
        return None

def generate_streaming_response(prompt, ollama_model="llama3.2"):
    """Streams the LLM response from Ollama API."""
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": ollama_model,
            "prompt": prompt,
            "stream": True
        }, stream=True, timeout=120)
        r.raise_for_status()

        full_response = ""
        for chunk in r.iter_lines():
            if chunk:
                try:
                    data = json.loads(chunk)
                    response_text = data.get("response", "")
                    full_response += response_text
                    yield response_text
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
        return full_response
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to LLM: Is Ollama running? Details: {e}")
        return "Error: Could not connect to LLM."


def get_final_prompt(incoming_query, context_json):
    """Constructs the final RAG system prompt with strict instructions."""
    
    prompt_template = f'''
You are **AIKARA**, a highly precise, professional AI Teaching Assistant specialized in the current video lecture material (Fuzzy Logic, Interface, and ML Techniques).

**YOUR CORE INSTRUCTIONS:**
1.  **Language:** Respond in the same language as the user's question (e.g., Hindi for Hindi, English for English).
2.  **Be Concise:** Answer the question directly and professionally.
3.  **STRICT CITATION RULE:** You MUST cite the source immediately following the relevant information. Use the **EXACT** format: **[Lecture Title or Number, TIME_IN_SECONDS]**. Example: [Lecture 2, 50.08].
4.  **NO EXTERNAL SOURCES:** Do NOT use or mention any external references, names, or books not explicitly present in the provided context (e.g., do not mention Zadeh or external publications).
5.  **NO PADDING:** Do not start or end with conversational padding (e.g., "You're interested in..." or "Does that help?").

**VIDEO SUBTITLE CONTEXT (For your use only):**
---
{context_json}
---
**USER QUESTION:** "{incoming_query}"

**AIKARA RESPONSE (Start immediately with the answer):**
'''
    return prompt_template

def cleanup_and_format_output(full_response_text):
    """Extracts citation, converts time, and cleans answer text."""
    
    # NEW REGEX: Matches the format [Fuzzy Set, 417.68] or [Lecture 2, 417.68]
    # Capture Groups: (1: Title/Number) (2: Time in Seconds)
    CITATION_PATTERN = r"\[\s*([^,\[\]]+),\s*([\d\.]+)\s*\]" 

    clean_answer = full_response_text.strip()
    citation = "Source: General Knowledge / Uncited"

    # Find the citation
    match = re.search(CITATION_PATTERN, full_response_text, flags=re.IGNORECASE)

    if match:
        lecture_title_or_num = match.group(1).strip()
        time_seconds = match.group(2)
        
        # Convert seconds to MM:SS format
        time_mm_ss = format_seconds_to_mm_ss(time_seconds)
        
        # Create the user-friendly citation string
        citation = f"Source: {lecture_title_or_num} | Time: {time_mm_ss} ({time_seconds}s)"
        
        # Remove all bracketed text (including citations and hallucinations) from the answer
        clean_answer = re.sub(r"\[.*?\]", '', full_response_text).strip()
        
        # Remove multi-line streaming artifacts and leading/trailing whitespace
        clean_answer = clean_answer.replace("\n", " ").strip()
        
        # FINAL SANITIZATION: Remove known conversational padding LLMs sometimes force
        clean_answer = re.sub(r"^(A fuzzy set is a mathematical concept used to represent uncertainty or imprecision in variables\s*)\.", "", clean_answer).strip()

    return clean_answer, citation

def process_and_stream_rag(incoming_query, df):
    """Executes the full RAG pipeline and streams the output."""
    
    # 1. RAG Retrieval
    question_embedding = create_embedding([incoming_query])
    if question_embedding is None:
        return "Error: Could not generate embedding."
        
    question_embedding = question_embedding[0]
    
    similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
    top_results = 5
    max_indx = similarities.argsort()[::-1][0:top_results]
    new_df = df.loc[max_indx] 
    
    context_json = new_df[["number", "title", "start", "text"]].to_json(orient="records")

    # 2. Create the prompt
    final_prompt = get_final_prompt(incoming_query, context_json)
    
    # 3. Stream the response
    return generate_streaming_response(final_prompt)


# --- UI IMPLEMENTATION ---

# Custom CSS for the AIKARA brand (Deep Dark Mode, Neon Accents, Sticky Input)
st.markdown("""
<style>
    /* Full-screen dark background */
    .stApp {
        background-color: #1a1a2e; /* Deep dark blue */
        color: #E0E0E0;
    }
    /* Hide the default Streamlit header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom Header Styling (AIKARA Logo Colors) */
    .aikara-header {
        text-align: center;
        padding: 30px 0 10px 0;
    }
    .aikara-title {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(to right, #3B82F6, #A855F7); /* Blue-Purple Gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 10px rgba(168, 85, 247, 0.5); /* Subtle Neon Glow */
    }
    
    /* Center the chat input and fix it to the bottom */
    .stChatInput {
        position: fixed;
        bottom: 0px;
        width: 100%;
        max-width: 900px; /* Constrain width similar to Gemini/ChatGPT */
        left: 50%;
        transform: translateX(-50%);
        padding: 10px 0;
        background-color: #1a1a2e;
        z-index: 100; /* Ensure it stays on top */
    }
    
    /* Style AI message bubble (Electric Blue) */
    .stChatMessage.stChatMessage--assistant {
        background-color: #2b3a60; /* Darker blue background for chat area */
        border-radius: 10px;
    }
    
    /* Source Citation Style */
    .source-citation {
        font-size: 0.8em;
        color: #A855F7; /* Neon Purple for source */
        margin-top: 5px;
        opacity: 0.8;
    }
    /* Ensure the chat history scrolls above the fixed input */
    section.main { padding-bottom: 100px; } 

</style>
""", unsafe_allow_html=True)


# --- AIKARA Header (Logo and Title) ---

st.markdown("""
<div class='aikara-header'>
    <h1 class='aikara-title'>AIKARA</h1>
    <p>AI-POWERED TEACHING ASSISTANT</p>
</div>
""", unsafe_allow_html=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            # Display source line below the AI bubble
            st.markdown(f'<p class="source-citation">{message["source"]}</p>', unsafe_allow_html=True)


# --- Chat Input & RAG Execution ---

if prompt := st.chat_input("Ask AIKARA a question (English or Hindi)..."):
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Prepare AI Response container for streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_text = ""
        
        # Execute RAG and stream output
        stream_generator = process_and_stream_rag(prompt, df)
        
        for chunk in stream_generator:
            full_response_text += chunk
            # Update the placeholder with the accumulating response text
            message_placeholder.markdown(full_response_text + "▌") # Use '▌' as a cursor
        
        # Remove cursor and finalize the text
        message_placeholder.markdown(full_response_text)
        
        # 3. Cleanup and Format
        clean_answer, citation = cleanup_and_format_output(full_response_text)
        
        # Overwrite the streamed placeholder with the final, clean answer
        message_placeholder.markdown(clean_answer)
        
        # Display the source below the main chat bubble
        st.markdown(f'<p class="source-citation">{citation}</p>', unsafe_allow_html=True)
        

    # 4. Save Final Output to History
    st.session_state.messages.append({
        "role": "assistant", 
        "content": clean_answer, 
        "source": citation
    })