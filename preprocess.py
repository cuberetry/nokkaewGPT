import os
import zipfile
from pythainlp.tokenize.tcc import segment
import json
from collections import Counter

def tk_segment(text):
    subwords = segment(text)

    if len(subwords) == 0:
        return ''

    text = []
    for word in subwords:
        if word.isdigit():
            continue
        else:
            text.append(word)

    return text


dir_path = "data"
data_path = []

if os.path.isdir(dir_path):
    file_names = os.listdir(dir_path)
    for file_name in file_names:
        file_name = "./data/" + file_name
        if zipfile.is_zipfile(file_name) and not os.path.exists(file_name[:-4]):
            with zipfile.ZipFile(file_name, 'r') as zip_ref:
                zip_ref.extractall("./data/")
        if file_name.split('.')[-1] in {"csv", "json"}:
            data_path.append(file_name)

output_path = "./data/subword_tokenize.txt"

with open('./data/subword_tokenize.txt', 'w') as reset_file:
    reset_file.write("")
for file in data_path:
    with open(file, 'r') as input_file, open('./data/subword_tokenize.txt', 'a') as subword_tokenize:
        print(file)
        if file == "./data/idx_to_word.json" or file == "./data/word_to_idx.json":
            continue
        if file.split('.')[-1] == "json":
            data = json.load(input_file)
            i = 0
            sum_key, con_key = None, None
            for d in data[0]:
                if d == "highlight" or d == "summary":
                    sum_key = d
                elif d == "content" or d == "text" or d == "body":
                    con_key = d
            for lines in data:
                if sum_key is None:
                    line = lines[con_key] + "\n"
                else:
                    line = lines[con_key] + '<h>' + lines[sum_key] + "\n"
                text = tk_segment(line)
                subword_tokenize.write(' '.join(text))
                if i % 100 == 0:
                    print(i)
                i += 1
        if file.split('.')[-1] == "csv":
            file_data = input_file.readlines()
            header = file_data[0].split(',')
            content_idx, summary_idx = None, None
            i = 0
            for h in range(len(header)):
                if header[h] == "body" or header[h] == "content" or header[h] == "text":
                    content_idx = h
                if header[h] == "summary" or header[h] == "highlight":
                    summary_idx = h
                if content_idx is not None and summary_idx is not None:
                    break
            # for line in file_data[1:]
            print(content_idx, summary_idx)
            for lines in file_data[1:]:
                line = lines.split(',')
                try:
                    p_line = line[content_idx] + '<h>' + line[summary_idx] + "\n"
                except IndexError:
                    pass
                text = tk_segment(p_line)
                subword_tokenize.write(' '.join(text))
                if i % 100 == 0:
                    print(i)
                i += 1

with open('./data/subword_tokenize.txt', 'r') as file:
    content = file.read()
    new_content = content.replace('   ', ' <s> ').replace('< h >', '<h>')

with open('./data/subword_tokenize.txt', 'w') as file:
    file.write(new_content)

print("Saving tokenized vocabs as dict")

# Tokenize
with open("./data/subword_tokenize.txt") as f:
    text = f.read()
words_tokens = text.split(' ')  # Change this to a list of sub-word instead

# Words and their frequency
word_cnt = Counter(words_tokens)
vocab = sorted(word_cnt, key=word_cnt.get, reverse=True)
# print(word_cnt)
vocab_size = len(vocab)

# Create indexing
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

# Save word_to_idx and idx_to_word as json
with open("./data/word_to_idx.json", 'w', encoding='utf-8') as word_to_idx_file:
    json.dump(word_to_idx, word_to_idx_file, ensure_ascii=False,)
print("Successfully saved word to index")

with open("./data/idx_to_word.json", 'w', encoding='utf-8') as idx_to_word_file:
    json.dump(idx_to_word, idx_to_word_file, ensure_ascii=False,)
print("Successfully saved index to word")
