import model.model as model
import torch

# Create a language model from the class
NokkaewGPT = model.NokkaewLanguageModel()
m = NokkaewGPT.to(model.device)

# Create an optimizer, adaptive learning rate algorithm
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# Train from scratch if reset is True
reset = False

total_step = int(input("Enter training step: "))

if not reset:
    model_state_dict = torch.load('./model/nokkaew_model.pth')
    NokkaewGPT.load_state_dict(model_state_dict)

for steps in range(total_step):
    # sample a batch of data
    xb, yb = model.get_batch('train')

    # evaluate the loss
    logits, loss = NokkaewGPT(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Progress report every 1000 steps
    if steps % 1000 == 0:
        print(str(steps) + '/' + str(total_step))

torch.save(NokkaewGPT.state_dict(), './model/nokkaew_model.pth')
