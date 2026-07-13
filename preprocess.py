# preprocess.py

import re
import numpy as np

# Lightweight stemmer to avoid an external dependency on NLTK.
# This is not a full Porter Stemmer but works for simple suffix stripping
# used by this small chatbot dataset.


# 1. Lowercase
def to_lowercase(text):
    return text.lower()


# 2. Tokenize
def tokenize(text):
    text = to_lowercase(text)
    tokens = re.findall(r"\b\w+\b", text)
    # normalize repeated letters (e.g., 'hii' -> 'hi')
    def normalize(word):
        # collapse 3+ repeats to 2, then remove duplicates beyond 1 for short words
        new = []
        prev = ''
        for ch in word:
            if ch == prev:
                # allow double letters, skip extras
                if len(new) >= 1 and new[-1] == ch:
                    continue
            new.append(ch)
            prev = ch
        return ''.join(new)

    return [normalize(t) for t in tokens]


# 3. Stemming
def stem(word):
    w = word.lower()
    # remove simple common suffixes
    for suf in ("ing", "ly", "ed", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 1:
            return w[:-len(suf)]
    return w


# 4. Build Vocabulary
def build_vocabulary(patterns):
    vocabulary = []

    for pattern in patterns:
        tokens = tokenize(pattern)

        for token in tokens:
            token = stem(token)

            if token not in vocabulary:
                vocabulary.append(token)

    vocabulary.sort()
    return vocabulary


# 5. Bag of Words
def bag_of_words(tokenized_sentence, vocabulary):
    sentence_words = [stem(word) for word in tokenized_sentence]

    bag = np.zeros(len(vocabulary), dtype=np.float32)

    for idx, word in enumerate(vocabulary):
        if word in sentence_words:
            bag[idx] = 1.0

    return bag