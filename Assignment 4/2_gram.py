
import string
import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

N_GRAM = 2 #defining n gram

#class for 2 gram
class TwoGram(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.names = [line.strip() for line in file.readlines()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        ngrams = []
        labels = []
        for i in range(len(name) - 2):
            ngram = name[i:i + 2]
            label = name[i + 2]
            ngrams.append(ngram)
            labels.append(label)
        return ngrams, labels

#class for 3 gram
class ThreeGram(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.names = [line.strip() for line in file.readlines()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        ngrams = []
        labels = []
        for i in range(len(name) - 3):
            ngram = name[i:i + 3]
            label = name[i + 3]
            ngrams.append(ngram)
            labels.append(label)
        return ngrams, labels

# Load the names from the file into a dataset
if N_GRAM == 3:
  dataset = ThreeGram('names.txt')
else:
  dataset = TwoGram('names.txt')

# Split the dataset into train, validation, and test sets
train_data, test_data = train_test_split(
    dataset, test_size=0.05, random_state=42)


train_data, val_data = train_test_split(
    train_data, test_size=(0.05/0.095), random_state=42)

#model class for 2 gram
class TwoGramFeedForwardNN(nn.Module):
    def __init__(self):
        super(TwoGramFeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(2 * 27, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 27)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

#model class for 3 gram
class ThreeGramFeedForwardNN(nn.Module):
    def __init__(self):
        super(ThreeGramFeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(3 * 27, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 27)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

batch_size = 32

#function for converting text dataset to onehot encoding
def collate_fn(batch):
    tensor_batch = []
    label_batch = []
    
    for ngrams, labels in batch:
        tensor = torch.zeros((len(ngrams), N_GRAM, 27))
        label = torch.zeros((len(ngrams), 27))
        
        for i, ngram in enumerate(ngrams):
            for j, char in enumerate(ngram):
                if char == '.':
                    char_idx = 26
                else:
                    char_idx = ord(char) - ord('a')
                tensor[i][j][char_idx] = 1
            
        for i, label_char in enumerate(labels):
            if label_char == ".":
              label_idx = 26
            else:  
              label_idx = ord(label_char) - ord('a')
            label[i][label_idx] = 1
            
        tensor_batch.append(tensor)
        label_batch.append(label)
        
    tensor_batch = torch.cat(tensor_batch)
    label_batch = torch.cat(label_batch)
    
    return tensor_batch, label_batch

train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(
    test_data, batch_size=batch_size, collate_fn=collate_fn)

if N_GRAM == 3:
  model = ThreeGramFeedForwardNN()
else:
  model = TwoGramFeedForwardNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
epoch_loss = []
perplexity = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = inputs.view(-1, N_GRAM * 27)  # Reshape the input tensor
        labels = labels.view(-1, 27)  # Reshape the label tensor

        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

        # Compute loss
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    avg_train_loss = running_loss/len(train_loader)
    epoch_loss.append(avg_train_loss)

    model.eval()
    valid_loss = 0.0
        
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
        
        # Compute average validation loss for the epoch
        avg_valid_loss = valid_loss / len(val_loader)
        
        # Print validation loss for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_valid_loss:.4f}")
        perplexity.append(math.e ** avg_valid_loss)
    # Print epoch statistics
    # epoch_loss = running_loss / len(train_loader)
    # epoch_accuracy = correct / total

    # print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2%}'.format(epoch +
    #       1, num_epochs, epoch_loss, epoch_accuracy))

#ploting dev set perplexity over epochs
plt.plot(perplexity)
plt.show()

#ploting traing set loss over epochs
plt.plot(epoch_loss)
plt.show()

#testing the model
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.view(-1, N_GRAM * 27)  # Reshape the input tensor
        labels = labels.view(-1, 27)  # Reshape the label tensor

        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

        # Compute loss
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        test_loss += loss.item()

# Print test statistics
test_loss /= len(test_loader)
test_accuracy = test_correct / test_total
print('Test Loss: {:.4f}, Test Accuracy: {:.2%}'.format(test_loss, test_accuracy))
print("Test perplexity: ", math.e ** test_loss)
