# Import necessary libraries for scraping, semantic analysis, and Streamlit UI
import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# Cache the embedding and keyword extraction models for reuse across reruns without reloading into memory
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, general-purpose sentence embedding model
    kw_model = KeyBERT(model)  # KeyBERT initialized with the above embedding model
    return model, kw_model

# Load the models into global variables for use in summarization and keyword extraction
model, kw_model = load_model()

# Extracts and validates the first URL from user input
def extract_url(text):
    match = re.search(r'(https?://\S+)', text)  # Regex to extract URL
    if match:
        url = match.group(1)
        # Ensure URL is from allowed domain for safety (SSRF protection)
        if url.startswith("https://www.churchofjesuschrist.org/"):
            return url
        else:
            return None  # Reject disallowed domains
    return None  # No URL found

# Scrape talk content (title, author, body) from the provided General Conference talk URL
@st.cache_data
def scrape_talk_content(url):
    page = requests.get(url)  # Fetch the page HTML
    soup = BeautifulSoup(page.content, 'html.parser')  # Parse with BeautifulSoup

    # Extract title
    title = soup.find('h1').text.strip() if soup.find('h1') else 'Title not found'
    # Extract author if available
    author_el = soup.find('p', class_='author-name')
    author = author_el.text.strip() if author_el else 'Author not found'

    # Extract and clean the main body content
    body = soup.find('div', class_='body-block')
    if body:
        # Remove image wrappers to clean the text
        for div in body.find_all('div', class_=re.compile(r'imageWrapper-wTPPD')):
            div.decompose()
        # Remove all hyperlinks to clean the text
        for a in body.find_all('a'):
            a.decompose()
        # Extract the cleaned text with line breaks
        content = body.get_text(separator='\n').strip()
    else:
        content = 'Content not found'

    # Return structured data for downstream use
    return {'Title': title, 'Author': author, 'Content': content}

# Summarizes the talk by selecting the top N semantically central paragraphs
def summarize_text_with_embeddings(text, num_paragraphs):
    # Split content into paragraphs and filter out trivial ones
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
    if not paragraphs:
        return "No sufficient content to summarize."

    # Encode each paragraph into a semantic embedding
    embeddings = model.encode(paragraphs, convert_to_tensor=True)
    # Compute the mean embedding to represent the talk's overall semantic content
    mean_emb = embeddings.mean(dim=0, keepdim=True)

    # Import cosine similarity function for scoring
    from sentence_transformers.util import cos_sim
    scores = cos_sim(embeddings, mean_emb).squeeze().tolist()

    # Select indices of top N paragraphs with highest semantic similarity to the mean
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_paragraphs]
    # Reorder the selected indices to match their original order for narrative coherence
    top_indices_sorted = sorted(top_indices)

    # Join the selected paragraphs with spacing for readability
    return '\n\n'.join([paragraphs[i] for i in top_indices_sorted])

# Extracts the top N keywords using KeyBERT with semantic similarity rather than pure frequency
def extract_top_keywords(text, top_n=3):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1,1),  # Unigrams only
        stop_words='english',         # Remove English stopwords
        top_n=top_n                   # Number of keywords to extract
    )
    # Capitalize for consistent display
    return [kw[0].capitalize() for kw in keywords]

# Main Streamlit app logic
def main():
    st.title("General Conference Semantic Summarizer")  # App title

    # Create an input form for URL and parameter selection
    with st.form("form"):
        # Header and Church website link styled with inline HTML
        st.markdown("""
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
            <p style='font-size: 20px; margin: 0;'>Paste Talk URL</p>
            <a href='https://www.churchofjesuschrist.org/study/general-conference?lang=eng' 
               target='_blank' 
               style='display: inline-block; padding: 0.4em 1.2em; font-size: 16px; font-weight: 500; color: white; 
                      background-color: #007bff; border-radius: 4px; text-decoration: none;'>
                Go to Church Website
            </a>
        </div>
        """, unsafe_allow_html=True)

        # User text input for the URL
        talk_url = st.text_input("")
        # User slider to select the number of paragraphs in the summary
        num_paragraphs = st.slider("Number of paragraphs:", 3, 10, 5)
        # Submit button for the form
        submitted = st.form_submit_button("Summarize")

    if submitted:
        # Extract URL safely from the user's input
        clean_url = extract_url(talk_url)
        if not clean_url:
            st.warning("Please enter a valid URL.")
            return

        # Scrape the content from the provided URL
        data = scrape_talk_content(clean_url)

        # Display talk title and author
        st.subheader(data['Title'])
        st.write(f"**{data['Author']}**")

        # Display extracted top keywords
        st.subheader("Top 3 Keywords")
        keywords = extract_top_keywords(data['Content'])
        for kw in keywords:
            st.write(f"• {kw}")

        # Display semantic summary
        st.subheader("Summary")
        summary = summarize_text_with_embeddings(data['Content'], num_paragraphs)
        st.write(summary)

        # Inform user about fair use of content
        st.info("Content used under fair use for personal study and summarization only.")

# Entry point for the Streamlit app
if __name__ == "__main__":
    main()
