import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model data, trained model, and intents data
with open('v1/intents.json', 'r') as f:
    intents = json.load(f)

FILE = "v1/data.pth"
data = torch.load(FILE)

input_size = data.get("input_size")
hidden_size = data.get("hidden_size")
output_size = data.get("output_size")
all_words = data.get("all_words")
tags = data.get("tags")
model_state = data.get("model_state")

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Chatbot Program
bot_name = "Usopp"
print("---USOPP CHATBOT---\nType 'quit' to end conversation.")

while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probabilities = torch.softmax(output, dim=1)
    probability = probabilities[0][predicted.item()]

    if probability.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["intent"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: What do you think?")
