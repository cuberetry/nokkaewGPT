from pythainlp.tokenize.tcc import segment
import model.model as model
import torch


def tk_segment(text):
    subwords = segment(text)
    subwords = ['<s>' if word == ' ' else word for word in subwords]

    if len(subwords) == 0:
        return ''

    text = []
    for word in subwords:
        if word.isdigit():
            continue
        else:
            text.append(word)
    return text

context = input("Enter your content: ")
tk_context = tk_segment(context)
encoded = model.encode(tk_context + ['<h>'])
encoded_t = torch.tensor(encoded)
encoded_t = encoded_t.unsqueeze(0)
endline = model.word_to_idx['<n>']

model_state_dict = torch.load('./model/nokkaew_model.pth')
NokkaewGPT = model.NokkaewLanguageModel()
model_state_dict['token_embedding_table.weight'] = torch.nn.Parameter(torch.randn(model.vocab_size, model.n_embd))
model_state_dict['lm_head.weight'] = torch.nn.Parameter(torch.randn(model.vocab_size, model.n_embd))
model_state_dict['lm_head.bias'] = torch.nn.Parameter(torch.randn(model.vocab_size))
NokkaewGPT.load_state_dict(model_state_dict)
m = NokkaewGPT.to(model.device)

with open("./output/output_from_model.txt", "w") as output_file:
    raw_output = m.generate(encoded_t.to(model.device), endline, 1000)[0].tolist()
    output_file.write(model.decode(raw_output))
