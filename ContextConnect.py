# Author: Sandeep Belgavi
# Created on: 2025-03-19

from flask import Flask, request, render_template, jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

class WordContextGame:
    def __init__(self, word_embeddings_file, target_word):
        self.word_embeddings = self.load_word_embeddings(word_embeddings_file)
        self.target_word = target_word.lower()
        if self.target_word not in self.word_embeddings:
            raise ValueError(f"Target word '{target_word}' not found in embeddings.")
        self.difficulty = 'medium'  # Default difficulty

    def set_difficulty(self, difficulty):
        """Sets the difficulty level and updates the target word."""
        if difficulty not in ['easy', 'medium', 'hard']:
            raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'.")
        self.difficulty = difficulty
        self.set_target_word()

    def set_target_word(self):
        """Sets the target word based on the difficulty level."""
        if self.difficulty == 'easy':
            self.target_word = 'cat'  # Example easy word
        elif self.difficulty == 'medium':
            self.target_word = 'king'  # Example medium word
        elif self.difficulty == 'hard':
            self.target_word = 'philosophy'  # Example hard word

    def calculate_similarity(self, guess_word):
        """Calculates cosine similarity between guess and target."""
        guess_word = guess_word.lower()
        if guess_word not in self.word_embeddings:
            return None  # Word not in vocabulary
        target_vector = self.word_embeddings[self.target_word]
        guess_vector = self.word_embeddings[guess_word]
        similarity = cosine_similarity([target_vector], [guess_vector])[0][0]
        return similarity

    def get_feedback(self, similarity):
        """Provides feedback based on the similarity score and difficulty level."""
        if self.difficulty == 'easy':
            if similarity >= 0.3:
                return "High similarity"
            elif similarity >= 0.1:
                return "Moderate similarity"
            else:
                return "Low similarity"
        elif self.difficulty == 'medium':
            if similarity >= 0.5:
                return "High similarity"
            elif similarity >= 0.2:
                return "Moderate similarity"
            else:
                return "Low similarity"
        elif self.difficulty == 'hard':
            if similarity >= 0.7:
                return "High similarity"
            elif similarity >= 0.4:
                return "Moderate similarity"
            else:
                return "Low similarity"

    def load_word_embeddings(self, file_path):
        """Loads word embeddings from a file (e.g., GloVe)."""
        word_embeddings = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                word_embeddings[word] = vector
        return word_embeddings

    def preprocess_text(self, text):
        """Preprocesses text for word similarity calculations."""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return tokens

    def calculate_similarity(self, guess_word):
        """Calculates cosine similarity between guess and target."""
        guess_word = guess_word.lower()
        if guess_word not in self.word_embeddings:
            return None  # Word not in vocabulary
        target_vector = self.word_embeddings[self.target_word]
        guess_vector = self.word_embeddings[guess_word]
        similarity = cosine_similarity([target_vector], [guess_vector])[0][0]
        return similarity

# Initialize the game
embeddings_file = "/Users/sandeep.b/IdeaProjects/Dpe-Agent/src/resources/glove.6B 2/glove.6B.300d.txt"  # Replace with your GloVe file path
target_word = "king"  # Choose a word
game = WordContextGame(embeddings_file, target_word)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/guess', methods=['POST'])
def guess():
    guess_word = request.form['guess']
    difficulty = request.form['difficulty']
    game.set_difficulty(difficulty)
    if guess_word.lower() == game.target_word:
        return jsonify(result="Congratulations! You guessed the word!", similarity=None)
    else:
        similarity = game.calculate_similarity(guess_word)
        if similarity is None:
            return jsonify(result="Word not in vocabulary. Try again.", similarity=None)
        else:
            feedback = game.get_feedback(similarity)
            return jsonify(result=f"Similarity: {similarity:.4f} - {feedback}", similarity=float(similarity))

if __name__ == "__main__":
    app.run(debug=True)