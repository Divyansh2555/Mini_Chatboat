import numpy as np
from dataload import load_data
from preprocess import tokenize, stem, build_vocabulary, bag_of_words

# Load data
intents = load_data()

all_patterns = []
all_tags = []
xy = []

# Read intents
for intent in intents["intents"]:
    tag = intent["tag"]
    all_tags.append(tag)

    for pattern in intent["patterns"]:
        tokens = tokenize(pattern)
        xy.append((tokens, tag))
        all_patterns.append(pattern)

# Build vocabulary
vocabulary = build_vocabulary(all_patterns)

print("Vocabulary:")
print(vocabulary)

# Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, vocabulary)
    X_train.append(bag)
    y_train.append(all_tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)