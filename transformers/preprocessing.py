def process():
    file = open('transformers/data/words.txt', 'r')
    data = file.read().split("\n")
    print(len(data))
    pig_latin = open('transformers/data/pig_latin.txt', 'w')
    i = 0
    for d in data:
        pig_latin.write(to_pig_latin(d) + '\n')


def to_pig_latin(word):
    vowels = ['a', 'e', 'i', 'o', 'u']
    if word[0] in vowels:
        word = word + 'way'
    else:
        i = get_syl_index(word)
        if i > 0:
            first_seg = word[0:i]
            second_seg = word[i:]
            word = second_seg + first_seg + 'ay'
    return word


def get_syl_index(word):
    vowels = ['a', 'e', 'i', 'o', 'u']
    for i, c in enumerate(word):
        if c in vowels:
            return i
    else:
        return -1


process()
