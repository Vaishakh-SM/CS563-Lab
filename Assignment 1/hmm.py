import json

json_data = []

with open('penn-data.json') as json_file:
    json_data = json.load(json_file)

THRESHOLD_FREQ = 4
RARE_TOKEN = "RARE"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
START_TAG = "<START>"
END_TAG = "<END>"

word_count = {}
for cnt, item in enumerate(json_data):
    json_data[cnt][0] = START_TOKEN + " " + json_data[cnt][0] + " " + END_TOKEN

    json_data[cnt][1].insert(0, START_TOKEN)
    json_data[cnt][1].append(END_TOKEN)

for item in json_data:
    for word in item[0].split(" "):
        if word in word_count:
            word_count[word] = word_count[word] + 1
        else:
            word_count[word] = 1

for cnt, item in enumerate(json_data):
    new_data = ""
    for word in item[0].split(" "):
        if word_count[word] <= THRESHOLD_FREQ:
            new_data = new_data + " " + RARE_TOKEN
        else:
            new_data = new_data + " " + word

    json_data[cnt][0] = new_data[1:]

word_count.clear()

for item in json_data:
    for word in item[0].split(" "):
        if word in word_count:
            word_count[word] = word_count[word] + 1
        else:
            word_count[word] = 1


def calc_transition(data):

    transition_count = {}
    word_id = {}
    word_cnt = 0

    for item in data:
        word_list = item[0].split(" ")
        for cnt, word in enumerate(word_list[:-1]):
            if word not in word_id:
                word_id[word] = word_cnt
                word_cnt += 1

            if word not in transition_count:
                transition_count[word] = {}

            next_word = word_list[cnt + 1]

            if next_word in transition_count[word]:
                transition_count[word][next_word] += 1
            else:
                transition_count[word][next_word] = 1

    return transition_count, word_id


def calc_emission(data):
    # Data expected to be in format of list of list. Inner list[0] = sentence, Inner list[1] = Tags
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

    # emission_count[RARE_TOKEN] = {"Most freq" : Rare count}

    return emission_count, tag_id, tag_from_id


def viterbi(sentence, emission_count, transition_count, word_count, tag_id,
            tag_from_id):
    word_list = sentence.split(" ")

    word_list.insert(0, START_TOKEN)
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
                if word_list[i] not in transition_count[word_list[i - 1]]:
                    t_count = 0
                else:
                    t_count = transition_count[word_list[i - 1]][word_list[i]]

                if j not in emission_count[word_list[i]]:
                    e_count = 0
                else:
                    e_count = emission_count[word_list[i]][j]

                prob = dp[i - 1][prev_tag] * (
                    t_count / word_count[word_list[i - 1]]) * (
                        e_count / word_count[word_list[i]])

                if prob > dp[i][j]:
                    dp[i][j] = prob
                    prev[i][j] = prev_tag

    max_prob = 0
    max_index = 0
    print("Dp is:")
    print(dp)

    print("Prev is")
    print(prev)
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

    decoded_tags

    return decoded_tags


def trial():
    trans_count, word_id = calc_transition(json_data)
    emission_count, tag_id, tag_from_id = calc_emission(json_data)
    sent = "Neither Lorillard nor the researchers who studied the workers were aware of any research on smokers of the Kent cigarettes."
    res = viterbi(sent, emission_count, trans_count, word_count, tag_id,
                  tag_from_id)

    return res
