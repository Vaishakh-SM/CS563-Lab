import os
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Define the paths to the train and test data files
train_path = '/home/muhammed_1901cs38/ML/NLP-assignment/aclImdb/test'
test_path = '/home/muhammed_1901cs38/ML/NLP-assignment/aclImdb/train'

# Define the spacy tokenizer
tokenizer = spacy.load('en_core_web_sm')

# Define the device to use for training
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
device = 'cpu'

# Define the hyperparameters
batch_size = 32
embedding_dim = 100
hidden_dim_1 = 256
hidden_dim_2 = 128
output_dim = 2
num_epochs = 10
learning_rate = 0.0005

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, index):
        # Convert text sequence to sequence of indices
        text = self.texts[index]
        if len(text) > self.max_len:
            text = text[:self.max_len]
        else:
            text += ['<PAD>'] * (self.max_len - len(text))
        text_indices = [self.vocab[token] if token in self.vocab else self.vocab['UNK'] for token in text]
        text_tensor = torch.LongTensor(text_indices)
        
        # Convert label to tensor
        label_tensor = torch.LongTensor([self.labels[index]])
        
        return text_tensor, label_tensor

limit=-1
c=0
# Read the train data
train_texts = []
train_labels = []
for label in ['pos', 'neg']:
    label_path = os.path.join(train_path, label)
    c=0
    for fname in os.listdir(label_path):
        c+=1
        if(c==limit):
            break
        if c%30==0:
           print(f'{c}\n')
        if fname.endswith('.txt'):
            with open(os.path.join(label_path, fname), 'r') as f:
                train_texts.append(f.read())
            train_labels.append(1 if label == 'pos' else 0)

c=0
# Read the test data
test_texts = []
test_labels = []
for label in ['pos', 'neg']:
    label_path = os.path.join(test_path, label)
    c=0
    for fname in os.listdir(label_path):
        c+=1
        if(c==limit):
            break
        if c%30==0:
            print(f'{c}\n')
        if fname.endswith('.txt'):
            with open(os.path.join(label_path, fname), 'r') as f:
                test_texts.append(f.read())
            test_labels.append(1 if label == 'pos' else 0)

# Tokenize the train texts
train_tokens = []
for text in train_texts:
    tokens = [tok.text for tok in tokenizer(text) if tok.is_alpha and not tok.is_stop]
    train_tokens.append(tokens)

# Tokenize the test texts
test_tokens = []
for text in test_texts:
    tokens = [tok.text for tok in tokenizer(text) if tok.is_alpha and not tok.is_stop]
    test_tokens.append(tokens)


# Assign "UNK" token to all words with frequency < 5
word_counter = Counter([token for tokens in train_tokens for token in tokens])
vocab = {'PAD': 0, 'UNK': 1}
for token, count in word_counter.items():
    if count >= 5:
        vocab[token] = len(vocab)

# Split the train data into a new training set and a validation set
train_tokens, valid_tokens, train_labels, valid_labels = train_test_split(train_tokens, train_labels, test_size=0.1)

max_len=140
# Create the datasets and data loaders
train_dataset = TextDataset(train_tokens, train_labels, vocab,max_len)
valid_dataset = TextDataset(valid_tokens, valid_labels, vocab,max_len)
test_dataset = TextDataset(test_tokens, test_labels, vocab,max_len)

#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class  TextClassifier(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, output_dim):
        super( TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x= self.embedding(x)
        x=x.permute(1,0,2)
        h0 = torch.zeros(1, x.size(1), self.hidden_dim)
        c0 = torch.zeros(1, x.size(1), self.hidden_dim)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[-1, :, :])
        out = self.softmax(out)
        
        return out

# max_len+=2
# Initialize the model and move it to the device
model = TextClassifier(len(vocab),embedding_dim, hidden_dim_1, output_dim)
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    train_loss = 0
    valid_loss = 0
    train_acc = 0
    valid_acc = 0
    train_preds = []
    train_targets = []
    valid_preds = []
    valid_targets = []
    model.train()
    
    for texts, labels in tqdm(train_loader):
        texts = texts.to(device)
        labels = labels.squeeze().to(device)
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (logits.argmax(dim=1) == labels).sum().item()
        preds=logits.argmax(dim=1)
        train_preds.extend(preds.tolist())
        train_targets.extend(labels.tolist())
    train_loss /= len(train_loader)
    train_acc /= len(train_dataset)
    model.eval()
    with torch.no_grad():
        for texts, labels in valid_loader:
            texts = texts.to(device)
            labels = labels.squeeze().to(device)
            logits = model(texts)
            loss = criterion(logits, labels)
            valid_loss += loss.item()
            valid_acc += (logits.argmax(dim=1) == labels).sum().item()
            preds=logits.argmax(dim=1)
            valid_preds.extend(preds.tolist())
            valid_targets.extend(labels.tolist())
        valid_loss /= len(valid_loader)
        valid_acc /= len(valid_dataset)
    valid_precision, valid_recall, valid_f1, _ = precision_recall_fscore_support(valid_targets, valid_preds, average='macro')
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_targets, train_preds, average='macro')
        
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print(f'Epoch {epoch + 1}:')
    print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}, Training Precision: {train_precision:.4f}, Training Recall: {train_recall:.4f}, Training F1-score: {train_f1:.4f}')
    print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.4f}, Validation Precision: {valid_precision:.4f}, Validation Recall: {valid_recall:.4f}, Validation F1-score: {valid_f1:.4f}')

# Evaluate the model on the test set
test_loss = 0
test_acc = 0
test_preds = []
test_targets = []
model.eval()
with torch.no_grad():
    for texts, labels in test_loader:
        texts = texts.to(device)
        labels = labels.squeeze().to(device)
        logits = model(texts)
        loss = criterion(logits, labels)
        test_loss += loss.item()
        test_acc += (logits.argmax(dim=1) == labels).sum().item()
        preds=logits.argmax(dim=1)
        test_preds.extend(preds.tolist())
        test_targets.extend(labels.tolist())

test_loss /= len(test_loader)
test_acc /= len(test_dataset)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='macro')
print(f'Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_acc:.4f}, Testing Precision: {test_precision:.4f}, Testing Recall: {test_recall:.4f}, Testing F1-score: {test_f1:.4f}')

