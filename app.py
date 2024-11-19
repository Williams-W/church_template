import streamlit as st
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to extract URL from text
def extract_url(text):
    # Regular expression pattern to match URLs
    url_pattern = r'(https?://\S+)'
    # Search for URLs in the text
    match = re.search(url_pattern, text)
    # If a URL is found, return it
    if match:
        return match.group(1)
    else:
        return None
    
# Function to scrape the content from a given talk URL
def scrape_talk_content(talk_URL):
    try:
        # Send a GET request to the URL
        page = requests.get(talk_URL)
        page.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML content
        soup = BeautifulSoup(page.content , 'html.parser')

        # Find the title element and extract its text content
        title_element = soup.find('h1')
        title_text = title_element.text.strip() if title_element else "Title not found"

        # Find the author element and extract its text content
        author_element = soup.find('p', class_='author-name')
        author_text = author_element.text.strip() if author_element else "Author not found"

        # Find the body element and extract its text content
        body_block_div = soup.find('div', class_='body-block')
        
        if body_block_div:
            # Remove any divs containing images
            for div in body_block_div.find_all('div', class_=re.compile(r'imageWrapper-wTPPD')):
                div.decompose()

            # Remove all <a> tags from the body content
            for a_tag in body_block_div.find_all('a'):
                a_tag.extract()

            # Extract the text content from the body
            content = body_block_div.get_text(separator='\n').strip()
        else:
            content = "Content not found"

        # Return the extracted information as a dictionary
        return {'Title': title_text, 'Author': author_text, 'Content': content}
    
    except RequestException as e:
        st.write("Please Check the URL")
        st.error(f"Error: {e}")
        return None

# Function to preprocess the text content
def preprocess_text(text):
    # Tokenize paragraphs
    paragraphs = text.split('\n\n')
    
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Preprocess each paragraph
    preprocessed_paragraphs = []
    for paragraph in paragraphs:
        # Tokenize words
        words = word_tokenize(paragraph)
        # Remove stopwords and lemmatize words
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
        # Join words back into paragraph
        preprocessed_paragraph = ' '.join(words)
        preprocessed_paragraphs.append(preprocessed_paragraph)
    
    return preprocessed_paragraphs

# Function to summarize text using TF-IDF
def summarize_text(text, num_paragraphs=7):  # Change num_paragraphs to 7
    # Tokenize paragraphs
    paragraphs = text.split('\n\n')

    # Preprocess text
    preprocessed_paragraphs = preprocess_text(text)
    
    # Convert preprocessed paragraphs to TF-IDF matrix
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(preprocessed_paragraphs)

    # Get vocabulary
    vocab = vectorizer.get_feature_names_out()

    # Calculate word scores
    word_scores = {}

    # Iterate over vocabulary
    for idx, word in enumerate(vocab):
        # Sum up TF-IDF scores for the word across all documents
        score = tfidf_matrix[:, idx].sum()
        # Store word score
        word_scores[word] = score

    # Filter only unigrams (words with no spaces)
    unigram_scores = {word: score for word, score in word_scores.items() if " " not in word}

    # Get top 4 most important unigrams
    top_four_words = sorted(unigram_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    
    # Calculate importance scores for paragraphs
    paragraph_scores = tfidf_matrix.sum(axis=1)
    
    # Get indices of top N paragraphs based on scores
    top_paragraph_indices = paragraph_scores.argsort(axis=0)[-num_paragraphs:].flatten()
    
    # Get top N original paragraphs
    top_paragraphs = [paragraphs[i] for i in range(len(paragraphs)) if i in top_paragraph_indices]
    
    # Join top paragraphs to form summary
    summary = '\n\n'.join(top_paragraphs)
    
    return summary, top_four_words

# Streamlit app
def main():
    st.title("General Conference Analysis")

    disclaimer_displayed = False  # Initialize the boolean variable

    # Using Streamlit forms to create the input field without "Press Enter to apply" message
    with st.form("talk_url_form"):

    # Positioning the link to the church website at the top-right corner
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
       
        talk_url = st.text_input("", key="talk_url_input")
        talk_url_stripped = extract_url(talk_url)
        submit_button = st.form_submit_button("Search")

    # Display disclaimer only if no search has been performed
    if not submit_button and not disclaimer_displayed:
        st.write("""Content sourced from The Church of Jesus Christ of Latter-day Saints is utilized solely for personal studies
        and lesson or talk preparation, in accordance with fair use principles. This usage is conducted independently and does not imply
        any ownership or claim of rights to the Church's materials by this site. By accessing and using this site, you agree to adhere to 
        all relevant guidelines governing the use of copyrighted material, including any terms of use provided by the Church.""")
        disclaimer_displayed = True  # Update the boolean variable

    # Button to trigger scraping
    if submit_button:
        if talk_url_stripped:
            # Scrape the content from the talk URL
            talk_content = scrape_talk_content(talk_url_stripped)

            if talk_content:
                st.subheader(talk_content['Title'])
                st.write(talk_content['Author'])

                # Three most important Words
                st.subheader("Most Frequent Words")
                summary, top_four_words = summarize_text(talk_content['Content'])  # Get top_four_words here
                top_words = [word[0].capitalize() for word in top_four_words]  # Extract and uppercase the words
                for word in top_words:
                    st.text(f"• {word}")  # Display each word as a bullet point

                # Summarize the text
                st.subheader("Summary")
                st.write(summary)

        else:
            st.warning("Please enter a valid URL.")

if __name__ == "__main__":
    main()
