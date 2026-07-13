import json

def load_data():
    with open("data/chat.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    return data