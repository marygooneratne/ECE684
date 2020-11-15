# import numpy as numpy
# import torch
# from torch import nn
# import torch.nn.functional as F
# import os

def gen_examples(input_file="data-mmg53/words.txt", output_file="data-mmg53/translated.txt"):
    input_data = []
    output_data = []
    for line in open(input_file, "r"): input_data.append(" ".join([c for c in line]))
    for line in open(output_file, "r"): output_data.append(" ".join([c for c in line]))
    examples = list(zip(input_data, output_data))
    for source, target in examples:
        print('Source: "{}", target: "{}"'.format(source, target))


def gen_files(length=10000, input_file="data-mmg53/words.txt", output_file="data-mmg53/translated.txt"):
    count = 0
    suffixes = ["ay", "way"]
    vowels = ["a", "e", "i", "o", "u"]
    with open(input_file, 'r') as input_file:
        with open(output_file, 'w') as output_file:
            for line in input_file:
                # if count > length:
                #     break
                for word in line.split():
                    word = word.lower()
                    consanants = ""
                    translated = ""
                    while word[0] not in vowels:
                        consanants = consanants + word[0]
                        if len(word) > 1:
                            word = word[1:]
                        else: 
                            word = ""
                            break
                    if len(consanants) > 0:
                        translated = word + consanants + suffixes[0]
                    else:
                        translated = word +suffixes[1]
                    output_file.write(translated+"\n")
                    count += 1
    input_file.close()
    output_file.close()


    
if __name__=="__main__":
    gen_files()
    gen_examples()
