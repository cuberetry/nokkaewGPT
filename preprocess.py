import os
import zipfile
from pythainlp.tokenize.tcc import segment
import json
import csv


def tk_segment(text):
    subwords = segment(text)

    if len(subwords) == 0:
        return ''
    elif len(subwords) == 1:
        return subwords[0]

    text = [subwords[0]]
    for word in subwords[1:]:
        if word.isdigit() and text[-1].isdigit():
            text[-1] += word
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
        if file.split('.')[-1] == "json":
            data = json.load(input_file)
            i = 0
            for lines in data:
                line = lines["content"] + '<h>' + lines["summary"] + "\n"
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
                if header[h] == "body":
                    content_idx = h
                if header[h] == "summary":
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
