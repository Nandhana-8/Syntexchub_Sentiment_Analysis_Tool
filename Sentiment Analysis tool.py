#Sentiment Analysis Tool
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
# Training Data
texts = [
    "I love this product",
    "Amazing experience",
    "Very happy with service",
    "Excellent quality",
    "Best item ever",
    "I hate this item",
    "Very bad experience",
    "Worst product",
    "Not good at all",
    "Terrible service"
]

labels = [
    "Positive", "Positive", "Positive", "Positive", "Positive",
    "Negative", "Negative", "Negative", "Negative", "Negative"
]

# Preprocess Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

cleaned_texts = [clean_text(t) for t in texts]

# Convert Text to Numeric Features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_texts)

# Train Model
model = MultinomialNB()
model.fit(X, labels)

# Evaluate Model
predictions = model.predict(X)
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, pos_label="Positive")

# CLI Interface
print("========================================")
print("     AI Sentiment Analysis Tool")
print("========================================")
print("Enter a review and detect sentiment")
print("Type 'exit' to close the program")
print("")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("F1 Score:", round(f1, 2))

while True:
    user_text = input("\nEnter text (type exit to stop) : ").strip()

    if user_text == "":
        print("Please enter some text.")
        continue

    if user_text.lower() == "exit":
        print("Program Closed.")
        break

    cleaned = clean_text(user_text)
    test = vectorizer.transform([cleaned])
    result = model.predict(test)[0]

    if result == "Positive":
        print("Predicted Sentiment : Positive Sentiment")
    else:
        print("Predicted Sentiment : Negative Sentiment")