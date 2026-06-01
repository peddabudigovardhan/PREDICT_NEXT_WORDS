import os
import warnings

# -------------------------------------------------------------------
# ০. এনভায়রনমেন্ট ও ওয়ার্নিং কনফিগারেশন (অবশ্যই সবার আগে থাকবে)
# -------------------------------------------------------------------
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # সব অপ্রয়োজনীয় লগ ও অ্যালার্ট বন্ধ করবে
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page configuration
st.set_page_config(
    page_title="Quote & Text AI Engine",
    page_icon="✍️",
    layout="wide"
)

# -------------------------------------------------------------------
# 1. CACHED FILE LOADERS
# -------------------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    """Loads and caches models and tokenizers to optimize performance."""
    assets = {}
    
    # Load Tokenizer
    if os.path.exists("tokenizer.pkl"):
        with open("tokenizer.pkl", "rb") as f:
            assets["tokenizer"] = pickle.load(f)
    else:
        st.error("Missing file: 'tokenizer.pkl'")
        
    # Load Max Length
    if os.path.exists("max_len.pkl"):
        with open("max_len.pkl", "rb") as f:
            assets["max_len"] = pickle.load(f)
    else:
        # Fallback default if pickle file is structure-dependent
        assets["max_len"] = 100 
        
    # Load Models
    if os.path.exists("lstm_model.h5"):
        assets["lstm"] = load_model("lstm_model.h5")
    if os.path.exists("rnn_model.h5"):
        assets["rnn"] = load_model("rnn_model.h5")
        
    return assets

@st.cache_data
def load_dataset():
    """Loads and caches the quote CSV file."""
    if os.path.exists("qoute_dataset.csv"):
        return pd.read_csv("qoute_dataset.csv")
    return None

# Initialize assets
assets = load_ml_assets()
df = load_dataset()

# -------------------------------------------------------------------
# 2. HELPER PREDICTION FUNCTION
# -------------------------------------------------------------------
def generate_text(seed_text, model, tokenizer, max_len, next_words=10):
    """Generates text sequentially using the chosen trained model architecture."""
    output_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_idx = np.argmax(predicted_probs, axis=-1)[0]
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_idx:
                output_word = word
                break
        if not output_word:
            break
        output_text += " " + output_word
    return output_text

# -------------------------------------------------------------------
# 3. USER INTERFACE LAYOUT
# -------------------------------------------------------------------
st.title("✍️ Quote & Text AI Studio")
st.caption("Deploying LSTM & RNN models built from your Jupyter notebook training workflows.")

# Setup functional Tabs
tab_inference, tab_data = st.tabs(["🚀 Model Inference", "📊 Dataset Explorer"])

# --- TAB 1: MODEL INFERENCE ---
with tab_inference:
    st.header("Generate Text Transitions")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        # Model Selection
        model_choice = st.selectbox(
            "Select Model Architecture:",
            options=["LSTM (lstm_model.h5)", "Vanilla RNN (rnn_model.h5)"]
        )
        
        # Word Length Controller
        words_to_gen = st.slider("Words to generate:", min_value=5, max_value=50, value=15)
        
        # Input Text Prompt
        user_prompt = st.text_input("Enter your starting phrase:", value="Life is short")
        
        # Button updated to avoid width warning if needed, but primary is fine.
        run_prediction = st.button("Generate Text Sequence", type="primary")

    with col2:
        st.subheader("Model Output")
        if run_prediction:
            selected_model_key = "lstm" if "LSTM" in model_choice else "rnn"
            
            # Validation Check
            if selected_model_key not in assets or "tokenizer" not in assets:
                st.error("The selected model or tokenizer assets could not be found in the current directory.")
            elif not user_prompt.strip():
                st.warning("Please enter a valid starting phrase.")
            else:
                with st.spinner(f"Processing sequence via {model_choice}..."):
                    try:
                        # Extract exact max_len parameter if it is saved as an integer or dict
                        raw_max_len = assets["max_len"]
                        max_len_int = raw_max_len if isinstance(raw_max_len, int) else int(list(raw_max_len)[0])
                        
                        generated_result = generate_text(
                            seed_text=user_prompt,
                            model=assets[selected_model_key],
                            tokenizer=assets["tokenizer"],
                            max_len=max_len_int,
                            next_words=words_to_gen
                        )
                        
                        # Display Results Elegantly
                        st.markdown("### Resulting Sequence:")
                        st.info(f'"{generated_result}"')
                    except Exception as e:
                        st.error(f"Execution Error during text processing: {e}")
        else:
            st.write("Adjust settings on the left panel and click 'Generate Text Sequence' to view model predictions.")

# --- TAB 2: DATASET EXPLORER ---
with tab_data:
    st.header("Dataset Inspection")
    if df is not None:
        st.success(f"Successfully loaded `qoute_dataset.csv` with {df.shape[0]} rows and {df.shape[1]} columns.")
        
        # Search Box
        search_query = st.text_input("Filter quotes by keyword:")
        filtered_df = df.copy()
        
        if search_query:
            # Dynamically look for common text columns
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                mask = df[text_cols[0]].astype(str).str.contains(search_query, case=False, na=False)
                filtered_df = df[mask]
        
        # Metrics Row
        m1, m2 = st.columns(2)
        m1.metric("Total Available Items", len(df))
        m2.metric("Filtered Matches", len(filtered_df))
        
        # Interactive Table Display - Updated parameter from use_container_width to width='stretch'
        st.subheader("Data Viewer")
        st.dataframe(filtered_df, width="stretch")
    else:
        st.warning("`qoute_dataset.csv` not detected in working root directory. Please upload or move your dataset file.")
