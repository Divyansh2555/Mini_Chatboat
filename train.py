import json

# chat.json को load करना
with open("data/chat.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# पूरा data print करना
print(data)

print("-" * 30)

# एक-एक question और answer print करना
for item in data:
    print("Question:", item["question"])
    print("Answer:", item["answer"])
    print()