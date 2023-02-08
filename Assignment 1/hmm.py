import json
import random
import math
from sklearn.metrics import classification_report

json_data = []

with open('penn-data.json') as json_file:
    json_data = json.load(json_file)

THRESHOLD_FREQ = 4
# Frequency after which a word is considered common or not rare.

RARE_TOKEN = "<RARE>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
START_TAG = "<START>"
END_TAG = "<END>"

# PREPROCESSING STEPS
w_cnt = {}
for cnt, item in enumerate(json_data):
    json_data[cnt][0] = START_TOKEN + " " + json_data[cnt][0] + " " + END_TOKEN

    json_data[cnt][1].insert(0, START_TOKEN)
    json_data[cnt][1].append(END_TOKEN)

for item in json_data:
    for word in item[0].split(" "):
        if word in w_cnt:
            w_cnt[word] = w_cnt[word] + 1
        else:
            w_cnt[word] = 1

for cnt, item in enumerate(json_data):
    new_data = ""
    for word in item[0].split(" "):
        if w_cnt[word] <= THRESHOLD_FREQ:
            new_data = new_data + " " + RARE_TOKEN
        else:
            new_data = new_data + " " + word

    json_data[cnt][0] = new_data[1:]

# PREPROCESSING DONE


def calc_transition(data):
    '''
    Calculates the transition counts between each tag.
    Takes in data as input, data is expected to be a list of lists where each
    item of the inner list has 2 elements sentence and tags.

    Returns a dictionary of transition counts of each tag to another

    To change the transition probability to trigram we have to consider
    transition from (state(i-2),state(i-1)) to state(i) rather than state(i-1) to state(i)
    '''
    transition_count = {}

    for item in data:
        tag_list = item[1]
        for cnt, tag in enumerate(tag_list[:-1]):

            if tag not in transition_count:
                transition_count[tag] = {}

            next_tag = tag_list[cnt + 1]

            if next_tag in transition_count[tag]:
                transition_count[tag][next_tag] += 1
            else:
                transition_count[tag][next_tag] = 1

    return transition_count


def calc_emission(data):
    '''
    Calculates the emission counts between a tag and a word.
    Takes in data as input, data is expected to be a list of lists where each
    item of the inner list has 2 elements sentence and tags.

    Returns a dictionary of emission counts of each word to a tag.
    '''
    emission_count = {}
    tag_id = {}
    tag_from_id = {}
    tag_cnt = 0

    for item in data:
        word_list = item[0].split(" ")

        for cnt, word in enumerate(word_list):
            if word not in emission_count:
                emission_count[word] = {}

            emitted_tag = item[1][cnt]
            if emitted_tag not in tag_id:
                tag_id[emitted_tag] = tag_cnt
                tag_from_id[tag_cnt] = emitted_tag
                tag_cnt += 1

            if tag_id[emitted_tag] not in emission_count[word]:
                emission_count[word][tag_id[emitted_tag]] = 1
            else:
                emission_count[word][tag_id[emitted_tag]] += 1

    return emission_count, tag_id, tag_from_id


def viterbi(sentence, emission_count, transition_count, tag_count, word_count,
            tag_id, tag_from_id):
    '''
    Performs viterbi algorithm
    Arguments:
    sentence: The sentence on which decoding is to be done
    emission_count : Dictionary of emission count
    transition_count: Dictionary of transition count
    tag_count: Dictionary of count of each tag in the training data
    word_count: Dictionary of count of each word in the training data
    tag_id: Mapping between tag and an integer
    tag_from_id: Mapping (dictionary) between tag_id and tag


    To change to trigram we need to calculate use the transition count
    from state(i-2),state(i-1)) to state(i) rather than state(i-1) to state(i)
    '''
    word_list = sentence.split(" ")
    if word_list[0] != START_TOKEN:
        word_list.insert(0, START_TOKEN)

    if word_list[-1] != END_TOKEN:
        word_list.append(END_TOKEN)
    new_list = []

    for word in word_list:
        if word not in word_count or word_count[word] <= THRESHOLD_FREQ:
            new_list.append(RARE_TOKEN)
        else:
            new_list.append(word)

    word_list = new_list

    prev = [[0] * len(tag_id) for i in range(len(word_list))]
    dp = [[0] * len(tag_id) for i in range(len(word_list))]

    dp[0][tag_id[START_TAG]] = 1
    prev[0][tag_id[START_TAG]] = -1

    for i in range(1, len(word_list)):
        for j in range(0, len(tag_id)):

            for prev_tag in range(0, len(tag_id)):

                t_count = transition_count.get(tag_from_id[prev_tag],
                                               {}).get(tag_from_id[j], 0)

                if j not in emission_count[word_list[i]]:
                    e_count = 0
                else:
                    e_count = emission_count[word_list[i]][j]

                prob = dp[i - 1][prev_tag] * (
                    t_count / tag_count[tag_from_id[prev_tag]]) * (
                        e_count / tag_count[tag_from_id[j]])

                if prob > dp[i][j]:
                    dp[i][j] = prob
                    prev[i][j] = prev_tag

    max_prob = 0
    max_index = 0

    for i in range(0, len(tag_id)):
        if dp[len(word_list) - 1][i] > max_prob:
            max_prob = dp[len(word_list) - 1][i]
            max_index = i

    curr = prev[len(word_list) - 1][max_index]
    decoded_seq = [curr]
    curr_word = len(word_list) - 1

    while curr_word > 0:

        curr = prev[curr_word - 1][curr]
        decoded_seq.append(curr)
        curr_word -= 1

    decoded_tags = []
    decoded_seq.reverse()

    for item in decoded_seq[1:]:
        decoded_tags.append(tag_from_id[item])

    return decoded_tags[1:]


def calc_tag_count(data):
    """
    Helper function to calculate tag counts
    """
    tag_count = {}

    for item in data:
        for tag in item[1]:
            if tag in tag_count:
                tag_count[tag] = tag_count[tag] + 1
            else:
                tag_count[tag] = 1

    return tag_count


def calc_word_count(data):
    """
    Helper function to calculate word counts
    """
    word_count = {}

    for item in data:
        for word in item[0].split(" "):
            if word in word_count:
                word_count[word] = word_count[word] + 1
            else:
                word_count[word] = 1

    return word_count


def calculate_difference(true_val, predicted_val):
    """
    Calculates the difference between predicted values and true values
    """
    assert len(true_val[1:-1]) == len(predicted_val)
    true_pred = 0
    false_pred = 0

    for cnt, value in enumerate(true_val[1:-1]):
        if predicted_val[cnt] != value:
            false_pred += 1
        else:
            true_pred += 1

    return true_pred, false_pred


def sample_testcase():
    sent = "The Soviet Union usually begins buying U.S. crops earlier in the fall."
    shuffled_data = random.sample(json_data, len(json_data))

    train_len = math.floor(len(shuffled_data) * 0.8)
    train = shuffled_data[0:train_len]
    test = shuffled_data[train_len + 1:]

    tag_count = calc_tag_count(train)
    word_count = calc_word_count(train)
    trans_count = calc_transition(train)
    emission_count, tag_id, tag_from_id = calc_emission(train)
    predicted_values = viterbi(sent, emission_count, trans_count, tag_count,
                               word_count, tag_id, tag_from_id)

    print("PRED: ", predicted_values)


def print_accuracy(flat_original_tags, flat_predicted_tags):
    assert len(flat_original_tags) == len(flat_predicted_tags)

    print("Accuracy report:")
    report = classification_report(flat_original_tags,
                                   flat_predicted_tags,
                                   zero_division=0)
    print(report)
    with open('./outputs/classification_report.txt', 'w') as outfile:
        print(report, file=outfile)


def main():
    shuffled_data = random.sample(json_data, len(json_data))

    train_len = math.floor(len(shuffled_data) * 0.8)
    print("Training", train_len, "samples...")
    train = shuffled_data[0:train_len]
    test = shuffled_data[train_len + 1:]

    word_count = calc_word_count(train)
    tag_count = calc_tag_count(train)
    trans_count = calc_transition(train)
    emission_count, tag_id, tag_from_id = calc_emission(train)
    total_true = 0
    total_false = 0
    flat_original_tags = []
    flat_predicted_tags = []
    predictions = []

    for item in test:
        sentence = item[0]
        true_value = item[1]
        predicted_values = viterbi(sentence, emission_count, trans_count,
                                   tag_count, word_count, tag_id, tag_from_id)

        new_pred = {
            "sentence": sentence[8:-6],
            "ground_truth": true_value[1:-1],
            "prediction": predicted_values
        }
        flat_original_tags.extend(true_value[1:-1])
        flat_predicted_tags.extend(predicted_values)

        predictions.append(new_pred)

        true_pred, false_pred = calculate_difference(true_value,
                                                     predicted_values)

        total_true += true_pred
        total_false += false_pred

    print_accuracy(flat_original_tags, flat_predicted_tags)

    print("Test results written!")
    json_object = json.dumps(predictions, indent=4)
    with open('./outputs/pred.json', 'w') as outfile:
        outfile.write(json_object)

    print("TRUE: ", total_true)
    print("FALSE: ", total_false)


if __name__ == "__main__":
    main()