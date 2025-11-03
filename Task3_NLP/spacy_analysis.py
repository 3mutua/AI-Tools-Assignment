import spacy
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt

print("Loading spaCy model...")
# Load English model
nlp = spacy.load("en_core_web_sm")

# Sample Amazon product reviews
reviews = [
    "I absolutely love my new iPhone 14 Pro from Apple. The camera quality is amazing and battery life lasts all day!",
    "This Samsung Galaxy phone is terrible. The screen cracked after one week and customer service was horrible.",
    "Google Pixel has the best Android experience with timely updates and clean interface.",
    "I bought a Sony headphones and the sound quality is incredible. Best purchase ever!",
    "My Dell laptop stopped working after 2 months. Worst product I've ever owned.",
    "Microsoft Surface Pro is perfect for work and entertainment. The pen is very responsive.",
    "This HP printer constantly jams and the ink is too expensive. Don't recommend.",
    "Apple MacBook Pro with M2 chip is blazing fast. Perfect for programming and design work.",
    "The Bose speakers have amazing sound quality but are quite expensive for what they offer.",
    "My Lenovo ThinkPad is reliable and durable. Great for business use."
]

print("Performing Named Entity Recognition and Sentiment Analysis...")

# Sentiment lexicon (basic rule-based approach)
positive_words = {
    'love', 'amazing', 'excellent', 'great', 'awesome', 'fantastic', 'perfect',
    'best', 'incredible', 'good', 'nice', 'wonderful', 'outstanding', 'superb',
    'brilliant', 'fast', 'responsive', 'reliable', 'durable'
}

negative_words = {
    'terrible', 'horrible', 'worst', 'bad', 'awful', 'poor', 'cheap', 'broken',
    'jams', 'expensive', 'cracked', 'stopped'
}

def analyze_sentiment(text):
    """Basic rule-based sentiment analysis"""
    doc = nlp(text.lower())
    positive_count = sum(1 for token in doc if token.text in positive_words)
    negative_count = sum(1 for token in doc if token.text in negative_words)
    
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

# Process each review
results = []

for i, review in enumerate(reviews):
    doc = nlp(review)
    
    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Filter for product-related entities
    product_entities = [ent for ent in entities if ent[1] in ['ORG', 'PRODUCT']]
    
    # Analyze sentiment
    sentiment = analyze_sentiment(review)
    
    # Store results
    results.append({
        'review_id': i+1,
        'review_text': review,
        'entities': product_entities,
        'sentiment': sentiment,
        'positive_words': [token.text for token in doc if token.text in positive_words],
        'negative_words': [token.text for token in doc if token.text in negative_words]
    })

# Create DataFrame for better visualization
df_results = pd.DataFrame(results)

print("\n=== Named Entity Recognition Results ===")
for result in results:
    print(f"\nReview {result['review_id']}:")
    print(f"Text: {result['review_text']}")
    print(f"Entities: {result['entities']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Positive words: {result['positive_words']}")
    print(f"Negative words: {result['negative_words']}")
    print("-" * 80)

# Summary statistics
print("\n=== Summary Analysis ===")
sentiment_counts = df_results['sentiment'].value_counts()
print("Sentiment Distribution:")
print(sentiment_counts)

# Extract all product entities
all_entities = []
for result in results:
    all_entities.extend([ent[0] for ent in result['entities']])

entity_counts = Counter(all_entities)
print(f"\nMost mentioned products/brands:")
for entity, count in entity_counts.most_common(10):
    print(f"  {entity}: {count} times")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Sentiment distribution
axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
           colors=['lightgreen', 'lightcoral', 'lightyellow'])
axes[0].set_title('Sentiment Distribution in Reviews')

# Top entities
top_entities = dict(entity_counts.most_common(8))
axes[1].barh(list(top_entities.keys()), list(top_entities.values()), color='skyblue')
axes[1].set_title('Top Mentioned Products/Brands')
axes[1].set_xlabel('Frequency')

plt.tight_layout()
plt.show()

# Display detailed entity analysis
print("\n=== Detailed Entity Analysis ===")
for entity_type in ['ORG', 'PRODUCT']:
    entities_of_type = [ent[0] for result in results for ent in result['entities'] if ent[1] == entity_type]
    if entities_of_type:
        print(f"\n{entity_type} entities found:")
        for entity, count in Counter(entities_of_type).most_common():
            print(f"  - {entity}: {count} mentions")