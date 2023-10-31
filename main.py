import os
import random

import numpy as np
import torch
import string

from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from torch import nn, optim
from transformers import BertTokenizer, BertModel
import networkx as nx
from collections import defaultdict
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

# 1. Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If you haven't downloaded the stopwords dataset, uncomment the next line
# nltk.download('stopwords')

# Define paths (you might need to adjust these)
DATA_PATH = "data/Inspec"
FILE_LIMIT = 10_000

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)


# # Create an instance of the lemmatizer
# lemmatizer = WordNetLemmatizer()
#
#
# def preprocess_text(text):
#     # Remove punctuation
#     text = text.translate(str.maketrans("", "", string.punctuation))
#
#     # Remove stopwords
#     stop_words = set(stopwords.words("english"))
#     text = " ".join([word for word in text.split() if word.lower() not in stop_words])
#
#     # Lemmatize the words (uncomment the next line to apply lemmatization)
#     # text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
#
#     return text
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# import random
#
# # Number of random words you want to pick
num_words = 50






def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    return text


def create_inspec_dataset_for_bert_and_graph():
    filelimit = FILE_LIMIT
    all_data = []
    combined_graph = nx.Graph()  # Initialize a combined graph

    # First pass: Read data and create the combined graph
    for filename in os.listdir(DATA_PATH):
        filelimit -= 1
        if filelimit == 0:
            break
        if filename.endswith(".abstr"):
            with open(os.path.join(DATA_PATH, filename), "r") as f:
                abstract = preprocess_text(f.read().strip())

            with open(
                    os.path.join(DATA_PATH, filename.replace(".abstr", ".uncontr")), "r"
            ) as f:
                keywords = f.read().strip().split("\n")

            all_data.append(
                {"filename": filename, "abstract": abstract, "keywords": keywords}
            )

            # Add to the combined graph
            graph = create_cooccurrence_graph(abstract)
            combined_graph = nx.compose(combined_graph, graph)

            # Draw the graph
            pos = nx.spring_layout(combined_graph)  # positions for all nodes

            # Set the background color of the entire figure to grey
            plt.figure(figsize=(10, 10), facecolor='grey')

            # Draw nodes with black color
            nx.draw_networkx_nodes(combined_graph, pos, node_color='black', node_size=500)

            # Draw edges with default colors and the specified width and alpha
            nx.draw_networkx_edges(combined_graph, pos, width=1.0, alpha=0.5)

            # Draw labels with white font color
            nx.draw_networkx_labels(combined_graph, pos, font_color='white', font_size=10)

            # Set the background color of the axis area to grey
            plt.gca().set_facecolor('grey')

            plt.title("Word Co-occurrence Graph")
            plt.axis('off')  # Turn off the axis for better visual
            plt.show()

    input("HELLOV2")

    # Train Node2Vec on the combined graph
    node2vec = Node2Vec(
        combined_graph, dimensions=64, walk_length=30, num_walks=200, workers=4
    )
    node2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Randomly select words from the node2vec vocabulary
    sample_words = random.sample(list(node2vec_model.wv.index_to_key), num_words)
    embeddings = [node2vec_model.wv[word] for word in sample_words]
    embeddings_matrix = np.vstack(embeddings)

    # 2. Dimensionality Reduction using t-SNE
    perplexity_value = len(embeddings_matrix) - 1

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
    reduced_embeddings = tsne.fit_transform(embeddings_matrix)
    
    # 3. Plotting
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(sample_words):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.annotate(
            word,
            xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )
    plt.show()

    input("HELLO")

    # Splitting the dataset
    train_data, temp_data = train_test_split(all_data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    def generate_embeddings(data_list):
        dataset = []
        for data in data_list:
            abstract = data["abstract"]
            keywords = data["keywords"]

            # Text Learning using BERT for abstract
            inputs_abstract = tokenizer(
                abstract, return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                outputs_abstract = model(**inputs_abstract.to(device))
            bert_embeddings_abstract = outputs_abstract.last_hidden_state
            averaged_bert_embeddings_abstract = torch.mean(bert_embeddings_abstract, dim=1)

            # Text Learning using BERT for keywords  <-- CHANGED
            inputs_keywords = tokenizer(
                ' '.join(keywords), return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                outputs_keywords = model(**inputs_keywords.to(device))
            bert_embeddings_keywords = outputs_keywords.last_hidden_state
            averaged_bert_embeddings_keywords = torch.mean(bert_embeddings_keywords, dim=1)

            # Create a graph for the current abstract
            current_graph = create_cooccurrence_graph(abstract)

            # Get graph embeddings using the trained Node2Vec model
            graph_embeddings = {}
            counter = 0
            for node in tokenizer.tokenize(abstract):
                if node in node2vec_model.wv:
                    graph_embeddings[node] = node2vec_model.wv[node]
                else:
                    counter += 1
                    graph_embeddings[node] = torch.zeros(64)

            # Ensure the graph embeddings are in the same order as the tokens
            token_order_embeddings = [
                torch.tensor(graph_embeddings[token])
                for token in tokenizer.tokenize(abstract)
            ]
            token_order_embeddings = torch.stack(token_order_embeddings)

            # Average the token_order_embeddings
            averaged_token_order_embeddings = torch.mean(
                token_order_embeddings, dim=0, keepdim=True
            )

            # Concatenate the embeddings (including keyword embeddings)  <-- CHANGED
            concatenated_embeddings = torch.cat(
                (averaged_bert_embeddings_abstract, averaged_token_order_embeddings, averaged_bert_embeddings_keywords),
                dim=-1
            )

            dataset.append(
                {
                    "filename": data["filename"],
                    "abstract": abstract,
                    "keywords": keywords,
                    "concatenated_embeddings": concatenated_embeddings,
                    "graph": current_graph,
                }
            )
        return dataset

    train_dataset = generate_embeddings(train_data)
    val_dataset = generate_embeddings(val_data)
    test_dataset = generate_embeddings(test_data)

    return train_dataset, val_dataset, test_dataset


def create_cooccurrence_graph(text, window_size=5):
    tokens = tokenizer.tokenize(text)
    graph = defaultdict(int)
    for i, token in enumerate(tokens):
        for j in range(i + 1, min(i + window_size, len(tokens))):
            pair = tuple(sorted([token, tokens[j]]))
            graph[pair] += 1
    G = nx.Graph()
    for (w1, w2), weight in graph.items():
        G.add_edge(w1, w2, weight=weight)
    return G


label_to_index = {"B": 0, "I": 1, "O": 2}
index_to_label = {0: "B", 1: "I", 2: "O"}


def labels_to_tensor(labels):
    return torch.tensor([label_to_index[label] for label in labels])


def label_keywords(text, keywords):
    tokens = tokenizer.tokenize(text)
    labels = ["O"] * len(tokens)

    for keyword in keywords:
        keyword_tokens = tokenizer.tokenize(keyword)
        for i in range(len(tokens) - len(keyword_tokens) + 1):
            if tokens[i : i + len(keyword_tokens)] == keyword_tokens:
                labels[i] = "B"
                for j in range(1, len(keyword_tokens)):
                    labels[i + j] = "I"
    return labels


# Placeholder for graph embedding techniques
def get_graph_embeddings(graph):
    # Create Node2Vec model
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

    # Train node2vec model
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Get embeddings for all nodes in the graph
    embeddings = {}
    for node in graph.nodes():
        embeddings[node] = model.wv[node]

    return embeddings


# Call the function to create the dataset
train_dataset, val_dataset, test_dataset = create_inspec_dataset_for_bert_and_graph()
print(f"TRAIN SET LENGTH: #{len(train_dataset)}")
print(f"VAL SET LENGTH: #{len(val_dataset)}")
print(f"TEST SET LENGTH: #{len(test_dataset)}")
# You can now use the 'train_dataset', 'val_dataset', and 'test_dataset' variables for further processing


class KeywordClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(KeywordClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Fully Connected layer
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)


missing_keys = [
    data["filename"] for data in train_dataset if "concatenated_embeddings" not in data
]
print(f"Number of entries missing 'concatenated_embeddings': {len(missing_keys)}")
if missing_keys:
    print("Filenames of missing entries:", missing_keys)

# Define the model, loss function, and optimizer
input_dim = train_dataset[0]["concatenated_embeddings"].size(-1)
print("INPUT DIM: " + str(input_dim))
output_dim = 3  # B, I, O

# Initialize the model without hidden_dim
model = KeywordClassifier(input_dim, output_dim).to(device)

# If you have multiple GPUs, wrap the model with nn.DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):  # Number of epochs
    model.train()
    for data in train_dataset:
        inputs = data["concatenated_embeddings"].unsqueeze(0).to(device)  # Add batch dimension
        if inputs.shape[1] != len(data["abstract"].split()):  # Check sequence length
            continue  # Skip this iteration if sequence length doesn't match

        labels = label_keywords(data["abstract"], data["keywords"])
        targets = labels_to_tensor(labels).unsqueeze(0).to(device)  # Add batch dimension

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs.view(-1, 3), targets.view(-1))
        loss.backward()
        optimizer.step()

    # Evaluation on validation set
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in val_dataset:
            inputs = data["concatenated_embeddings"].unsqueeze(0).to(device)  # Add batch dimension
            labels = label_keywords(data["abstract"], data["keywords"])
            targets = labels_to_tensor(labels).unsqueeze(0).to(device)  # Add batch dimension

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 2)
            total += targets.size(1)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}, Validation Accuracy: {accuracy:.2f}%")

print("TESTING BEGINS")
print("_______________________________________")

    # Evaluation on test dataset
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for data in test_dataset:
        inputs = data["concatenated_embeddings"].unsqueeze(0).to(device)
        labels = label_keywords(data["abstract"], data["keywords"])
        targets = labels_to_tensor(labels).unsqueeze(0).to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 2)
        total += targets.size(1)
        correct += (predicted == targets).sum().item()
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


# ## CLASSIC ML CLASSIFIER
#
#
# # Convert the dataset into features and labels for training the Random Forest classifier
# def prepare_dataset_for_rf(dataset):
#     X = []
#     y = []
#     for data in dataset:
#         # Using concatenated embeddings as features
#         features = data["concatenated_embeddings"].numpy().flatten()
#         X.append(features)
#
#         # For simplicity, we'll use the first label of each abstract as the target label
#         # This is a simplification and in a real-world scenario, you might want to handle labels differently
#         labels = label_keywords(data["abstract"], data["keywords"])
#         y.append(label_to_index[labels[0]])
#     return np.array(X), np.array(y)
#
#
# # Prepare datasets
# X_train, y_train = prepare_dataset_for_rf(train_dataset)
# X_val, y_val = prepare_dataset_for_rf(val_dataset)
#
# # Train a Random Forest classifier
# clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
# clf.fit(X_train, y_train)
#
# # Predict on validation set
# y_pred = clf.predict(X_val)
#
# # Calculate accuracy
# accuracy = accuracy_score(y_val, y_pred)
# print(f"Validation Accuracy: {accuracy:.2f}%")
