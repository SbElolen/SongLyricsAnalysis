# Code to analyse what people sing, and their lyrics, if it's a happy or sad one.

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# To download necessary resources|Do note that even when I tried this process first time, it gave me so much trouble too
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset gotten
df = pd.read_csv('song_lyrics.csv')


def preprocessed_text(text):
    """This cleans and tokenizes lyrics"""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    return ' '.join(words)


df['Cleaned_Lyrics'] = df['Lyrics'].apply(preprocessed_text)

# Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()
df['Sentiment'] = df['Cleaned_Lyrics'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['Sentiment_Label'] = df['Sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Sentiment Distribution Visualization
sns.countplot(x='Sentiment_Label', data=df, palette='coolwarm')
plt.title('Sentiment Distribution of Song Lyrics')
plt.show()

# To Generate WordCloud for the Positive and Negative Lyrics, if available
positive_text = ' '.join(df[df['Sentiment_Label'] == 'Positive']['Cleaned_Lyrics'])
negative_text = ' '.join(df[df['Sentiment_Label'] == 'Negative']['Cleaned_Lyrics'])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(WordCloud(width=400, height=200, background_color='white').generate(positive_text))
plt.axis('off')
plt.title('Positive Lyrics WordCloud')

plt.subplot(1, 2, 2)
plt.imshow(WordCloud(width=400, height=200, background_color='black').generate(negative_text))
plt.axis('off')
plt.title('Negative Lyrics WordCloud')

plt.show()

# Emotion Analysis using NRC Lexicon (Requires NRC dataset, load as needed)

# Example approach (assuming a dictionary mapping words to emotions)
# df['Emotion'] = df['Cleaned_Lyrics'].apply(lambda x: detect_emotion(x))
# sns.countplot(x='Emotion', data=df, palette='viridis')
# plt.title('Emotion Analysis of Song Lyrics')
# plt.show()

print("Analysis complete!")
