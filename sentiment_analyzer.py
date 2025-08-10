from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
reviews = [
    "I loved this movie, it was amazing!",
    "What a fantastic performance!",
    "Absolutely wonderful experience",
    "The plot was boring and predictable",
    "Worst movie I have ever seen",
    "I hated this film, it was terrible",
]

labels = ["positive", "positive", "positive", "negative", "negative", "negative"]

# Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

# Train the model
model = MultinomialNB()
model.fit(X, labels)

# Test the model with new input
while True:
    review = input("\nEnter a movie review (or 'quit' to exit): ")
    if review.lower() == "quit":
        break
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    print(f"Sentiment: {prediction[0]}")
