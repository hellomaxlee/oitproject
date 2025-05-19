import streamlit as st
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from openai import OpenAI
import urllib.parse, requests

# Your key
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# GitHub info
GITHUB_USER = "hellomaxlee"
GITHUB_REPO = "math356"
BRANCH = "main"
BASE_RAW = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}"

president_folders = {
    "Reagan": "President Reagan",
    "Obama": "President Obama",
    "Trump": "President Trump"
}
president_years = {
    "Reagan": range(1981, 1989),
    "Obama": range(2009, 2017),
    "Trump": range(2017, 2021)
}

@st.cache_data
def generate_files(folder, years):
    files = []
    for year in years:
        files.append(f"{folder}/State_of_the_Union_{year}.txt")
        files.append(f"{folder}/Inaugural_Address_{year}.txt")
        files.append(f"{folder}/Farewell_Address_{year}.txt")
    return files

@st.cache_data
def download_and_tokenize(file_paths):
    text = ""
    for file in file_paths:
        encoded_path = urllib.parse.quote(file)
        url = f"{BASE_RAW}/{encoded_path}"
        response = requests.get(url)
        if response.status_code == 200:
            text += response.text + " "
    return [simple_preprocess(text)]

def confidence_level(num_words_found):
    return min(num_words_found, 5)

def confidence_color(level):
    colors = {
        1: "#b71c1c",  # dark red
        2: "#c62828",  # crimson
        3: "#d84315",  # dark orange
        4: "#ef6c00",  # amber orange
        5: "#f9a825",  # yellow-gold
        6: "#afb42b",  # olive
        7: "#558b2f",  # moss green
        8: "#2e7d32",  # forest green
        9: "#1b5e20",  # darker green
        10: "#004d40"  # deep teal
    }
    return colors.get(min(level, 10), "#424242")  # fallback: charcoal

# Load everything
tokenized_data = {p: download_and_tokenize(generate_files(f, president_years[p]))
                  for p, f in president_folders.items()}

models = {p: Word2Vec(sentences=tokenized_data[p], vector_size=50, window=5, min_count=1, sg=1, epochs=100)
          for p in tokenized_data}

def get_gpt_interpretation(president, keyword, similar_words):
    prompt = f"""
You are an expert in political speech analysis.

The user is studying how U.S. presidents use the word '{keyword}'.

For President {president}, the 5 most similar words based on Word2Vec are: {', '.join(similar_words)}.

Please explain what this suggests about the rhetorical/historical context and thematic framing of '{keyword}' in {president}'s speeches. Reference specific moments during the presidency to contextualize word choice. Use 2-3 sentences and stick to how the words connect to facts while being convincing. However, if the connection is tenuous, don't say anything rather than trying to make a connection. In those cases, the explanation can be less than 2-3 sentences. Make sure your response for each president aligns with a president's ideology (for example, don't indicate that Trump loves immigration.)
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Streamlit app
import numpy as np

st.title("🏛 Presidential Word2Vec Explorer")
st.write("Explore how Reagan, Obama, and Trump frame different concepts in their speeches! This app was created using text data from Inauguration, Farewell, and State of the Union addresses and the Word2Vec skipgram algorithm. The five most similar words for each president are words likely to appear in a similar context as the input word.")

user_word = st.text_input("Enter a word (e.g., economy, democracy, race):").strip().lower()

if user_word:
    for pres, model in models.items():
        st.subheader(f"🗣️ {pres}")
        if user_word in model.wv:
            similar = model.wv.most_similar(user_word, topn=5)
            
            # Display similar words with scores
            for word, score in similar:
                st.write(f"**{word}** — _similarity score_: {score:.4f}")

            # Count actual appearances of the input word in tokenized data
            token_list = tokenized_data[pres][0]  # get token list from single-item list
            word_count = token_list.count(user_word)

            # Confidence score based on frequency: 1 if seen once, up to 10 if seen 10+ times
            confidence_level = min(word_count, 10)

            color = confidence_color(confidence_level)
            words = [w for w, _ in similar]
            interpretation = get_gpt_interpretation(pres, user_word, words)

            # Show interpretation with color and confidence
            st.markdown(
                f"<div style='background-color:{color}; color:white; padding:10px; border-radius:5px'>"
                f"<strong>Interpretation Confidence: {confidence_level}/10</strong><br>{interpretation}</div>",
                unsafe_allow_html=True
            )
        else:
            st.warning(f"⚠️ '{user_word}' not found in {pres}'s vocabulary.")

