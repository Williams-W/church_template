import streamlit as st
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import re
from sentence_transformers import SentenceTransformer, util
import nltk

# --- One-time NLTK setup ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Load embedding model with caching ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

# --- URL extraction ---
def extract_url(text):
    url_pattern = r'(https?://\S+)'
    match = re.search(url_pattern, text)
    return match.group(1) if match else None

# --- Scrape talk content ---
@st.cache_data
def scrape_talk_content(talk_URL):
    try:
        page = requests.get(talk_URL)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, 'html.parser')

        title_element = soup.find('h1')
        title_text = title_element.text.strip() if title_element else "Title not found"

        author_element = soup.find('p', class_='author-name')
        author_text = author_element.text.strip() if author_element else "Author not found"

        body_block_div = soup.find('div', class_='body-block')
        if body_block_div:
            for div in body_block_div.find_all('div', class_=re.compile(r'imageWrapper-wTPPD')):
                div.decompose()
            for a_tag in body_block_div.find_all('a'):
                a_tag.extract()
            content = body_block_div.get_text(separator='\n').strip()
        else:
            content = "Content not found"

        return {'Title': title_text, 'Author': author_text, 'Content': content}

    except RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return None

# --- Semantic summarization ---
def summarize_text_with_embeddings(text, num_paragraphs):
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]  # Filter short noise
    if not paragraphs:
        return "No sufficient content to summarize."

    embeddings = model.encode(paragraphs, convert_to_tensor=True)
    mean_embedding = embeddings.mean(dim=0, keepdim=True)
    cosine_scores = util.cos_sim(embeddings, mean_embedding).squeeze().tolist()

    top_indices = sorted(range(len(cosine_scores)), key=lambda i: cosine_scores[i], reverse=True)[:num_paragraphs]
    top_paragraphs = [paragraphs[i] for i in top_indices]

    return '\n\n'.join(top_paragraphs)

# --- Streamlit App ---
def main():
    st.title("General Conference Semantic Summarizer")

    with st.form("talk_url_form"):
        st.markdown("""
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <p style='font-size: 20px;'>Paste Talk URL</p>
            <a href='https://www.churchofjesuschrist.org/study/general-conference?lang=eng' target='_blank' style='background-color: #007bff; color: white; padding: 0.3em 0.8em; border-radius: 4px; text-decoration: none;'>Go to Church Website</a>
        </div>
        """, unsafe_allow_html=True)

        talk_url = st.text_input("")
        num_paragraphs = st.slider("Number of paragraphs to summarize:", min_value=3, max_value=10, value=5)
        submit = st.form_submit_button("Summarize")

    if submit:
        talk_url_clean = extract_url(talk_url)
        if not talk_url_clean:
            st.warning("Please enter a valid URL.")
            return

        talk_content = scrape_talk_content(talk_url_clean)
        if not talk_content:
            st.error("Failed to fetch talk content.")
            return

        st.subheader(talk_content['Title'])
        st.write(f"**{talk_content['Author']}**")

        st.subheader("Summary")
        summary = summarize_text_with_embeddings(talk_content['Content'], num_paragraphs)
        st.write(summary)

        st.info("Content used under fair use for personal study and summarization only.")

if __name__ == "__main__":
    main()
