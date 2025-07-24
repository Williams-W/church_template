import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    kw_model = KeyBERT(model)
    return model, kw_model

model, kw_model = load_model()

def extract_url(text):
    match = re.search(r'(https?://\S+)', text)
    return match.group(1) if match else None

@st.cache_data
def scrape_talk_content(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    title = soup.find('h1').text.strip() if soup.find('h1') else 'Title not found'
    author_el = soup.find('p', class_='author-name')
    author = author_el.text.strip() if author_el else 'Author not found'
    body = soup.find('div', class_='body-block')
    if body:
        for div in body.find_all('div', class_=re.compile(r'imageWrapper-wTPPD')):
            div.decompose()
        for a in body.find_all('a'):
            a.decompose()
        content = body.get_text(separator='\n').strip()
    else:
        content = 'Content not found'
    return {'Title': title, 'Author': author, 'Content': content}

def summarize_text_with_embeddings(text, num_paragraphs):
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
    if not paragraphs:
        return "No sufficient content to summarize."
    embeddings = model.encode(paragraphs, convert_to_tensor=True)
    mean_emb = embeddings.mean(dim=0, keepdim=True)
    from sentence_transformers.util import cos_sim
    scores = cos_sim(embeddings, mean_emb).squeeze().tolist()
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_paragraphs]
    return '\n\n'.join([paragraphs[i] for i in top_indices])

def extract_top_keywords(text, top_n=3):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,1), stop_words='english', top_n=top_n)
    return [kw[0].capitalize() for kw in keywords]

def main():
    st.markdown("""<style>div.stSlider > div[data-baseweb='slider'] > div > div {background: #007bff;} button {background-color: #007bff !important; color: white !important;} a {background-color: #007bff !important; color: white !important;} </style>""", unsafe_allow_html=True)
    st.title("General Conference Semantic Summarizer")
    with st.form("form"):
        st.markdown("""<div style='display: flex; justify-content: space-between; align-items: center;'>
        <p style='font-size: 20px;'>Paste Talk URL</p>
        <a href='https://www.churchofjesuschrist.org/study/general-conference?lang=eng' target='_blank' style='padding: 0.3em 0.8em; border-radius: 4px; text-decoration: none;'>Go to Church Website</a>
        </div>""", unsafe_allow_html=True)
        talk_url = st.text_input("")
        num_paragraphs = st.slider("Number of paragraphs:", 3, 10, 5)
        submitted = st.form_submit_button("Summarize")
    if submitted:
        clean_url = extract_url(talk_url)
        if not clean_url:
            st.warning("Please enter a valid URL.")
            return
        data = scrape_talk_content(clean_url)
        st.subheader(data['Title'])
        st.write(f"**{data['Author']}**")
        st.subheader("Top 3 Keywords")
        keywords = extract_top_keywords(data['Content'])
        for kw in keywords:
            st.write(f"• {kw}")
        st.subheader("Summary")
        summary = summarize_text_with_embeddings(data['Content'], num_paragraphs)
        st.write(summary)
        st.info("Content used under fair use for personal study and summarization only.")

if __name__ == "__main__":
    main()
