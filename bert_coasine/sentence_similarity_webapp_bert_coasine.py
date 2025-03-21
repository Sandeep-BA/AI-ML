# Python
from flask import Flask, request, jsonify, render_template_string
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Author: Sandeep Belgavi
# Date: 2023-10-05

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    # Get the output from the BERT model
    outputs = model(**inputs)
    # Return the mean of the last hidden state as the sentence embedding
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

@app.route('/')
def index():
    # Render the HTML form for inputting sentences
    return render_template_string('''
        <html>
        <head>
            <style>
                body {
                    background-color: #f0f8ff;
                    color: #333;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                }
                h1 {
                    margin-top: 50px;
                    color: #2c3e50;
                    text-align: center;
                }
                form {
                    margin: 20px auto;
                    padding: 20px;
                    border: 1px solid #ccc;
                    background-color: #fff;
                    width: 300px;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }
                input[type="text"] {
                    width: calc(100% - 20px);
                    padding: 10px;
                    margin: 10px 0;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                }
                .button {
                    margin-top: 10px;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    background-color: #3498db;
                    color: white;
                    cursor: pointer;
                    font-size: 16px;
                }
                .button:hover {
                    background-color: #2980b9;
                }
                table {
                    margin: 20px auto;
                    border-collapse: collapse;
                    width: 80%;
                    table-layout: fixed;
                    word-wrap: break-word;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 10px;
                }
                th {
                    background-color: #3498db;
                    color: white;
                }
                td {
                    background-color: #f9f9f9;
                }
                .explanation {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    width: 300px;
                    height: 400px;
                    overflow-y: scroll;
                    padding: 20px;
                    border: 1px solid #ccc;
                    background-color: #fff;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }
                .explanation img {
                    width: 100%;
                    height: auto;
                }
                .collapsible {
                    background-color: #3498db;
                    color: white;
                    cursor: pointer;
                    padding: 10px;
                    width: 100%;
                    border: none;
                    text-align: left;
                    outline: none;
                    font-size: 16px;
                }
                .content {
                    padding: 0 18px;
                    display: none;
                    overflow: hidden;
                    background-color: #f9f9f9;
                }
                .scrollable {
                    max-height: 200px;
                    overflow-y: scroll;
                }
            </style>
        </head>
        <body>
            <h1> Bert Model Embeddings Generator along with Cosine Similarity Score</h1>
            <form id="similarityForm">
                <label for="sentence1">Sentence 1:</label><br>
                <input type="text" name="sentence1" id="sentence1"><br><br>
                <label for="sentence2">Sentence 2:</label><br>
                <input type="text" name="sentence2" id="sentence2"><br><br>
                <input type="submit" value="Submit" class="button">
                <input type="button" value="Clear" class="button" onclick="clearForm()">
            </form>
            <table id="similarityTable" class="scrollable">
                <thead>
                    <tr>
                        <th>Similarity Score</th>
                    </tr>
                </thead>
                <tbody>
                </tbody>
            </table>
            <table id="embeddingTable" class="scrollable">
                <thead>
                    <tr>
                        <th>Sentence 1</th>
                        <th>Sentence 2</th>
                    </tr>
                </thead>
                <tbody>
                </tbody>
            </table>
            <button type="button" class="collapsible">How It Works</button>
            <div class="content">
                <h2>How It Works</h2>
                <p>This application uses a pre-trained BERT model to generate embeddings for input sentences. Here is how it works:</p>
                <ol>
                    <li>The BERT model and tokenizer are loaded using the <code>transformers</code> library.</li>
                    <li>When you submit two sentences, they are tokenized and passed through the BERT model to generate embeddings.</li>
                    <li>The embeddings are the mean of the last hidden state of the BERT model.</li>
                    <li>The cosine similarity between the two embeddings is calculated to determine how similar the sentences are.</li>
                    <li>The similarity score and embeddings are displayed on the web page.</li>
                </ol>
            </div>
            <script>
                document.getElementById('similarityForm').onsubmit = async function(event) {
                    event.preventDefault();
                    const sentence1 = document.getElementById('sentence1').value;
                    const sentence2 = document.getElementById('sentence2').value;
                    const response = await fetch('/similarity', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ sentence1, sentence2 })
                    });
                    const result = await response.json();
                    document.getElementById('similarityTable').querySelector('tbody').innerHTML = '<tr><td>' + result.similarity + '</td></tr>';
                    document.getElementById('embeddingTable').querySelector('tbody').innerHTML = '<tr><td class="scrollable">' + JSON.stringify(result.embedding1) + '</td><td class="scrollable">' + JSON.stringify(result.embedding2) + '</td></tr>';
                };

                function clearForm() {
                    document.getElementById('sentence1').value = '';
                    document.getElementById('sentence2').value = '';
                    document.getElementById('similarityTable').querySelector('tbody').innerHTML = '';
                    document.getElementById('embeddingTable').querySelector('tbody').innerHTML = '';
                }

                var coll = document.getElementsByClassName("collapsible");
                for (var i = 0; i < coll.length; i++) {
                    coll[i].addEventListener("click", function() {
                        this.classList.toggle("active");
                        var content = this.nextElementSibling;
                        if (content.style.display === "block") {
                            content.style.display = "none";
                        } else {
                            content.style.display = "block";
                        }
                    });
                }
            </script>
        </body>
        </html>
    ''')
@app.route('/embed', methods=['POST'])
def embed():
    # Handle embedding generation for a single sentence
    data = request.json
    sentence = data.get('sentence', '')
    embedding = get_embedding(sentence)
    return jsonify({'embedding': embedding.tolist()})

@app.route('/similarity', methods=['POST'])
def similarity():
    # Handle similarity calculation between two sentences
    data = request.json
    sentence1 = data.get('sentence1', '')
    sentence2 = data.get('sentence2', '')

    embedding1 = get_embedding(sentence1)
    embedding2 = get_embedding(sentence2)

    similarity_score = cosine_similarity(embedding1, embedding2)[0][0]
    return jsonify({'similarity': float(similarity_score), 'embedding1': embedding1.tolist(), 'embedding2': embedding2.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)