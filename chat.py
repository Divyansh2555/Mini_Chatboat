
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataload import load_data
from preprocess import tokenize, stem, build_vocabulary, bag_of_words
from dataset import ChatDataset
from model import NeuralNet


def build_training_data(intents):
	all_patterns = []
	all_tags = []
	xy = []

	for intent in intents["intents"]:
		tag = intent["tag"]
		if tag not in all_tags:
			all_tags.append(tag)

		for pattern in intent["patterns"]:
			tokens = tokenize(pattern)
			xy.append((tokens, tag))
			all_patterns.append(pattern)

	vocabulary = build_vocabulary(all_patterns)

	X_train = []
	y_train = []

	for (pattern_sentence, tag) in xy:
		bag = bag_of_words(pattern_sentence, vocabulary)
		X_train.append(bag)
		y_train.append(all_tags.index(tag))

	X_train = np.array(X_train, dtype=np.float32)
	y_train = np.array(y_train, dtype=np.longlong)

	return vocabulary, all_tags, X_train, y_train


def train_and_save(intents, epochs=200, batch_size=8, lr=0.001, hidden_size=8, checkpoint_path="checkpoints/chatbot.pth"):
	vocabulary, tags, X_train, y_train = build_training_data(intents)

	input_size = len(vocabulary)
	output_size = len(tags)

	dataset = ChatDataset(X_train, y_train)
	train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = NeuralNet(input_size, hidden_size, output_size).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	for epoch in range(1, epochs + 1):
		for (words, labels) in train_loader:
			words = words.to(device)
			labels = labels.to(device)

			outputs = model(words)
			loss = criterion(outputs, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if epoch % 50 == 0 or epoch == 1:
			print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

	# Save checkpoint
	data = {
		"model_state": model.state_dict(),
		"input_size": input_size,
		"hidden_size": hidden_size,
		"output_size": output_size,
		"vocabulary": vocabulary,
		"tags": tags
	}

	os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
	torch.save(data, checkpoint_path)
	print(f"Model saved to {checkpoint_path}")

	return checkpoint_path


def load_checkpoint(path="checkpoints/chatbot.pth"):
	if not os.path.exists(path):
		return None

	data = torch.load(path, map_location=torch.device('cpu'))
	input_size = data["input_size"]
	hidden_size = data["hidden_size"]
	output_size = data["output_size"]
	vocabulary = data["vocabulary"]
	tags = data["tags"]

	model = NeuralNet(input_size, hidden_size, output_size)
	model.load_state_dict(data["model_state"])
	model.eval()

	return model, vocabulary, tags


def chat_loop(model, vocabulary, tags, intents, device=torch.device('cpu'), threshold=0.45):
	print("Start chatting (type 'quit' or 'exit' to stop)")

	while True:
		sentence = input("You: ")
		if sentence.lower() in ("quit", "exit"):
			print("Goodbye!")
			break

		tokens = tokenize(sentence)
		X = bag_of_words(tokens, vocabulary)
		X = torch.from_numpy(X).unsqueeze(0).to(device)

		with torch.no_grad():
			outputs = model(X)
			probs = torch.softmax(outputs, dim=1)
			prob, predicted = torch.max(probs, dim=1)

			tag = tags[predicted.item()]

			if prob.item() > threshold:
				for intent in intents["intents"]:
					if intent["tag"] == tag:
						print("Bot:", np.random.choice(intent["responses"]))
						break
			else:
				print("Bot: I'm not sure I understand. Can you rephrase?")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
	parser.add_argument("--hidden", type=int, default=16, help="Hidden layer size")
	parser.add_argument("--batch", type=int, default=8, help="Batch size")
	parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
	parser.add_argument("--test", action="store_true", help="Run a short train test and exit")
	parser.add_argument("--no-chat", action="store_true", help="Do not start interactive chat after training")
	parser.add_argument("--retrain", action="store_true", help="Retrain model even if checkpoint exists")
	parser.add_argument("--threshold", type=float, default=0.35, help="Prediction probability threshold")
	args = parser.parse_args()

	intents = load_data()
	checkpoint = "checkpoints/chatbot.pth"

	loaded = load_checkpoint(checkpoint)
	if loaded is None or args.retrain:
		print("Training a new model...")
		# reduce epochs for quick test
		epochs = 2 if args.test else args.epochs
		train_and_save(intents, epochs=epochs, batch_size=args.batch, lr=args.lr, hidden_size=args.hidden, checkpoint_path=checkpoint)
		if args.test:
			print("Test training complete.")
			return

	model, vocabulary, tags = load_checkpoint(checkpoint)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	if args.no_chat:
		print("Model loaded. Exiting because --no-chat was specified.")
		return

	chat_loop(model, vocabulary, tags, intents, device=device, threshold=args.threshold)


if __name__ == '__main__':
	main()

