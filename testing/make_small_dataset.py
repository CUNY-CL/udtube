import conllu
import glob
import os

# TODO remove file after PR review
# Tl;dr I put a couple dev files into the dataset folder and just shortened each to 10 sentences and renamed them to toy
print(os.getcwd())
for file in glob.iglob('datasets/*.conllu', recursive=True):
    out_file = file.replace('dev', 'toy')
    with open(file, 'r') as input_file, open(out_file, 'w') as sink:
        sentences = conllu.parse_incr(input_file)
        for i, sentence in enumerate(sentences):
            if i == 10:
                break
            else:
                sink.writelines([sentence.serialize() + "\n"])

