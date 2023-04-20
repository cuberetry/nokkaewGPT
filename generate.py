from pythainlp.tokenize.tcc import segment
import model.model as model
import torch


def tk_segment(text):
    sub_words = segment(text)

    if len(sub_words) == 0:
        return ''
    elif len(sub_words) == 1:
        return sub_words[0]

    text = [sub_words[0]]
    for word in sub_words[1:]:
        if word.isdigit() and text[-1].isdigit():
            text[-1] += word
        else:
            text.append(word)

    return text


context = input("Enter your content: ")
tk_context = tk_segment(context)
encoded = model.encode(tk_context + ['<h>'])
encoded_t = torch.tensor(encoded)
encoded_t = encoded_t.unsqueeze(0)
endline = model.encode('\n')[0]

model_state_dict = torch.load('./model/nokkaew_model.pth')
NokkaewGPT = model.NokkaewLanguageModel()
NokkaewGPT.load_state_dict(model_state_dict)
m = NokkaewGPT.to(model.device)

with open("./output/output_from_model.txt", "w") as output_file:
    output_file.write(model.decode(m.generate(encoded_t, endline)[0].tolist()))
