import nltk
import spacy
from newsapi import NewsApiClient
from transformers import pipeline
from collections import Counter

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# Enter your API Key here
API_KEY = "YOUR_API_KEY_HERE"

newsapi = NewsApiClient(api_key=API_KEY)

# Load NLP Models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")

def fetch_news(query="current affairs"):
    articles = newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=5)
    return articles['articles']

def clean_text(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return tokens

def extract_keywords(text):
    tokens = clean_text(text)
    freq = Counter(tokens)
    return freq.most_common(10)

def analyze_article(article):
    content = article['content']
    if content:
        summary = summarizer(content, max_length=120, min_length=50, do_sample=False)[0]['summary_text']
        sentiment = sentiment_analyzer(content)[0]
        keywords = extract_keywords(content)

        return summary, sentiment, keywords
    return None, None, None

def main():
    print("\n🧠 SMART CURRENT AFFAIRS ANALYZER\n")
    query = input("Enter topic (e.g., RBI, ISRO, Elections, Budget): ")

    articles = fetch_news(query)

    for i, article in enumerate(articles, 1):
        print(f"\n🔹 NEWS {i}: {article['title']}")

        summary, sentiment, keywords = analyze_article(article)

        print("\n📌 Summary:")
        print(summary)

        print("\n😊 Sentiment:")
        print(sentiment)

        print("\n🔑 Keywords:")
        print(keywords)

if __name__ == "__main__":
    main()