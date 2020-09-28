
import sys
from decimal import *
import codecs


def viterbi_algorithm(obs, tags, transition_prob, emission_prob, tag_count, word_set):
    # print "In Viterbi\n"
    global tag_set
    current_prob = {}
    for tag in tag_map.keys():
        tp = Decimal(0)
        em = Decimal(0)
        if A["start"][tag] > 0:
            tp = Decimal(A["start"][tag])
        if obs[0].lower() in word_map.keys:
            if (obs[0].lower()+"/"+tag) in emission_prob:
                em = Decimal(emission_prob[word_list[0].lower()+"/"+tag])
                current_prob[tag] = tp * em
        else:
            em = Decimal(1) / (tag_count[tag] + len(word_set))
            current_prob[tag] = tp

    if len(word_list) == 1:
        max_path = max(current_prob, key=current_prob.get)
        return max_path
    else:
        for i in xrange(1, len(word_list)):
            previous_prob = current_prob
            current_prob = {}
            locals()['dict{}'.format(i)] = {}
            previous_tag = ""
            for tag in tags:
                if word_list[i].lower() in word_set:
                    if word_list[i].lower()+"/"+tag in emission_prob:
                        em = Decimal(
                            emission_prob[word_list[i].lower()+"/"+tag])
                        max_prob, previous_state = max((Decimal(previous_prob[previous_tag]) * Decimal(
                            transition_prob[previous_tag + "~tag~" + tag]) * em, previous_tag) for previous_tag in previous_prob)
                        current_prob[tag] = max_prob
                        locals()['dict{}'.format(i)
                                 ][previous_state + "~" + tag] = max_prob
                        previous_tag = previous_state
                else:
                    em = Decimal(1) / (tag_count[tag] + len(word_set))
                    max_prob, previous_state = max((Decimal(previous_prob[previous_tag]) * Decimal(
                        transition_prob[previous_tag+"~tag~"+tag]) * em, previous_tag) for previous_tag in previous_prob)
                    current_prob[tag] = max_prob
                    locals()['dict{}'.format(i)
                             ][previous_state + "~" + tag] = max_prob
                    previous_tag = previous_state

            if i == len(word_list)-1:
                max_path = ""
                last_tag = max(current_prob, key=current_prob.get)
                max_path = max_path + last_tag + " " + previous_tag
                for j in range(len(word_list)-1, 0, -1):
                    for key in locals()['dict{}'.format(j)]:
                        data = key.split("~")
                        if data[-1] == previous_tag:
                            max_path = max_path + " " + data[0]
                            previous_tag = data[0]
                            break
                result = max_path.split()
                result.reverse()
                return " ".join(result)
