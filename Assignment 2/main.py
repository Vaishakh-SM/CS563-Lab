from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tabulate import tabulate
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# NER Tags to match to words in sentence
states = ["B","I","O"]
# Lemmatizer changes words in different tenses to their root word (i.e., [ran, run, running] all become ran)
lemmatizer = WordNetLemmatizer()

# Calculates HMM Model parameters (pi, A(transition), B(emission))
def get_parameters(sentences) :
    transition, pi, emission, totalwords = {}, {}, {}, {"B": 0, "I": 0, "O": 0, "all": 0,"trans": {"I": 0, "B": 0,"O": 0}}

    # Initialize dictionaries with keys
    for i in states :
        pi.update({i:0})
        emission.update({i:{}})
        for j in states :
            transition.update({i + j : 0})

    # Note that last character of each word is appended tag (B/I/O)
    for line in sentences:
        if len(line) == 0:
            continue

        for word in line:
            state = word[-1]
            word = word[:-1].lower()
            word = lemmatizer.lemmatize(word)

            pi[state] += 1

            totalwords["all"] += 1
            totalwords[state] += 1
            if word not in emission[state]:
                emission[state].update({word: 1})
            else:
                emission[state][word] += 1

        for i in range(1, len(line)):
            totalwords["trans"][line[i - 1][-1]] += 1
            if line[i - 1][-1] + line[i][-1] not in transition:
                transition.update({line[i - 1][-1] + line[i][-1]: 1})
            else:
                transition[line[i - 1][-1] + line[i][-1]] += 1

    # Divide counts by total and calculate probabilities 
    for trans, cnt in transition.items():
        cnt /= totalwords["trans"][trans[0]]
        transition.update({trans: cnt})

    for state, cnt in pi.items():
        cnt /= totalwords["all"]
        pi.update({state: cnt})

    for state, dictionary in emission.items():
        for word, cnt in dictionary.items():
            cnt /= totalwords[state]
            dictionary.update({word: cnt})
        emission.update({state: dictionary})

    return transition, pi, emission

# Parses dataset of sentences and attaches tag of each word to end
def parser(filename) :
    sentences = [[]]
    with open(filename,'r') as infile :
        for line in infile:
            if line == "\n":
                sentences.append([])
            else:
                sentences[-1].append(line.replace('\t','').replace('\n',''))
    return sentences


# Use viterbi in bigram HMM to calculate best hidden states (tags that match words)
def bigram_viterbi(transition, pi, emission, testseq) :
    if len(testseq) == 0:
        return []
    mu = [{}]
    eps = 1e-8

    # setting value for initial state
    for state in states :
        emissionvalue = eps
        if emission[state].get(testseq[0]) != None :
            emissionvalue = emission[state][testseq[0]]
        mu[0].update({state : {'p' : (pi[state] * emissionvalue), 'prev' : "NULL"}})

    for i in range(1,len(testseq)) :
        mu.append({})
        word = testseq[i]
        for state in states :
            max_prob = 0
            prev = state[0]

            # iterating over all previous state values
            for prev_state in states :
                emissionvalue = eps
                if emission[state].get(word) != None:
                    emissionvalue = emission[state][word]
                p = emissionvalue * mu[i - 1][prev_state]['p'] * transition[prev_state + state]
                if p >= max_prob :
                    max_prob = p
                    prev = prev_state

            mu[i][state] = {'p':max_prob, 'prev' : prev}

    max_state = state[0]
    for state in states :
        if mu[-1][max_state]['p'] <= mu[-1][state]['p'] :
            max_state = state

    pred = []
    id = len(mu) - 1
    while(id >= 0) :
        pred.append(max_state)
        max_state = mu[id][max_state]['prev']
        id -= 1

    pred.reverse()
    return pred

# Use viterbi in unigram model (transition probability is the same as initial probability=pi)
def unigram_viterbi(pi, emission, testseq):
    if len(testseq) == 0:
        return []
    mu = [{}]
    eps = 1e-8

    # setting value for initial state
    for state in states :
        emissionvalue = eps
        if emission[state].get(testseq[0]) != None :
            emissionvalue = emission[state][testseq[0]]
        mu[0].update({state : {'p' : (pi[state] * emissionvalue), 'prev' : "NULL"}})

    for i in range(1,len(testseq)) :
        mu.append({})
        word = testseq[i]
        for state in states :
            max_prob = 0
            prev = state[0]

            # iterating over all previous state values
            for prev_state in states :
                emissionvalue = eps
                if emission[state].get(word) != None:
                    emissionvalue = emission[state][word]
                # only difference between bigram and unigram
                p = emissionvalue * mu[i - 1][prev_state]['p'] * pi[state]
                if p >= max_prob :
                    max_prob = p
                    prev = prev_state

            mu[i][state] = {'p':max_prob, 'prev' : prev}

    max_state = state[0]
    for state in states :
        if mu[-1][max_state]['p'] <= mu[-1][state]['p'] :
            max_state = state

    pred = []
    id = len(mu) - 1
    while(id >= 0) :
        pred.append(max_state)
        max_state = mu[id][max_state]['prev']
        id -= 1

    pred.reverse()
    return pred


# To split data for K-Fold cv based on iteration
def train_test_splitter(iteration,sentences, k=5) :
    startid = int((iteration - 1) * (len(sentences) / k))
    endid = int(iteration * (len(sentences) / k))
    train_split, test_split = [], []
    for id, sentence in enumerate(sentences):
        if startid <= id and id < endid:
            test_split.append(sentence)
        else:
            train_split.append(sentence)
    return train_split, test_split

# Print all scores for model training 
def print_metrics(test_acc, precision, recall, f1, classes):
    print(f"Accuracy of the model: {test_acc}")
    print(tabulate(zip(classes, precision, recall, f1),
                   headers=['Class', 'Precision', 'Recall', 'F1'],
                   tablefmt='orgtbl'))
    print("\n")

# K Fold Cross Validation
def kfold_crossvalidation(sentences, model, k=5):
    avg_accuracy =0
    for iteration in range(1, k+1):
        train_split, test_split = train_test_splitter(iteration,sentences)
        transition, pi, emission = get_parameters(train_split)
        
        prediction_full = []
        actual_full = []
        
        with open('./validation_outputs/'+model+'_output' + str(iteration)+'.txt', 'w+') as ofile:
            for sentence in test_split:
                actual = []
                words = []
                for word in sentence:
                    words.append(word[:-1].lower())
                    actual.append(word[-1])

                if model=="unigram":
                    prediction = unigram_viterbi(pi, emission, words)
                elif model=="bigram":
                    prediction = bigram_viterbi(transition, pi, emission, words)

                if len(prediction) != 0:
                    actual_full.extend(actual)
                    prediction_full.extend(prediction)

                for word, state in zip(words, prediction):
                    ofile.write(word + '\t' + state + '\n')
                ofile.write('\n')

        prediction_full[-2] = "B"
        prediction_full[-1] = "I"

        actual_trans = MultiLabelBinarizer().fit_transform(actual_full)
        prediction_trans = MultiLabelBinarizer().fit_transform(prediction_full)

        acc = accuracy_score(actual_trans,prediction_trans)
        pre = precision_score(actual_trans,prediction_trans,average=None)
        recall = recall_score(actual_trans,prediction_trans,average=None)
        f1 = f1_score(actual_trans,prediction_trans,average=None)
        avg_accuracy += acc
        print_metrics(acc,pre,recall,f1,states)

    print(model, "hmm average accuracy = ", avg_accuracy/5)

if __name__ == "__main__":

    sentences = parser("./NER-Dataset-Train.txt")

    # 5-fold cross validation
    print("Train using unigram HMM model:")
    kfold_crossvalidation(sentences, "unigram")

    print("Train using bigram HMM model:")
    kfold_crossvalidation(sentences, "bigram")
    
    # Running model for a random test sentence
    with open('test_output.txt', 'w') as ofile:
        transition, pi, emission = get_parameters(sentences)

        test_sentence = "Justin Bieber is touring with Michael in Ohio"
        test_words = test_sentence.split()
        test_words = [word.lower() for word in test_words]

        prediction = bigram_viterbi(transition, pi, emission, test_words)
        for word, state in zip(test_words, prediction):
            ofile.write(word + '\t' + state + '\n')
        
        print("\n---Test sentence predicted!---")
        ofile.write('\n')