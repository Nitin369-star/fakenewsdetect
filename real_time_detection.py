import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline

# Use NewsAPI or any news API to get the latest news
API_KEY = 'your_newsapi_key'  # Replace with your NewsAPI key
BASE_URL = 'https://newsapi.org/v2/top-headlines'

def get_news():
    params = {
        'country': 'us',  # Change to any country code
        'apiKey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        news_data = response.json()
        return news_data['articles']
    else:
        print("Error fetching news")
        return []

def process_article(article):
    title = article['title']
    url = article['url']
    content = article.get('content', 'No content available')
    return title, url, content

def analyze_news(news_data):
    # Using Hugging Face transformers for sentiment analysis
    sentiment_analyzer = pipeline('sentiment-analysis')
    results = []
    for article in news_data:
        title, url, content = process_article(article)
        sentiment = sentiment_analyzer(content)
        results.append({
            'title': title,
            'url': url,
            'sentiment': sentiment[0]['label'],
            'content': content
        })
    return pd.DataFrame(results)

def detect_emotional_manipulation(sentiment):
    if sentiment == "POSITIVE":
        return "Possible Manipulation: Emotional appeal"
    elif sentiment == "NEGATIVE":
        return "Possible Manipulation: Emotional distress"
    else:
        return "Neutral: No emotional manipulation detected"

def main():
    news_data = get_news()
    if news_data:
        df = analyze_news(news_data)
        df['emotional_manipulation'] = df['sentiment'].apply(detect_emotional_manipulation)
        print(df)
        df.to_csv('news_analysis.csv', index=False)  # Save the results to CSV for later use
    else:
        print("No news data available.")

if __name__ == "__main__":
    main()
