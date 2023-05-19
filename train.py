import model.model as model
import torch
from datetime import datetime

# Create a language model from the class
NokkaewGPT = model.NokkaewLanguageModel()
m = NokkaewGPT.to(model.device)
best_loss = None

# Create an optimizer, adaptive learning rate algorithm
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# Train from scratch if reset is True
reset = False
if input("Do you want to reset the model? [Enter 'yes' for reset]: ").upper() == "YES":
    reset = True
    print("Resetting...")

try:
    total_step = int(input("Enter training step: "))
except ValueError:
    print("Invalid input. Please enter a positive integer.")
    exit(1)
    
if total_step < 0:
    print("Invalid input. Please enter a positive integer.")
    exit(1)

if not reset:
    model_state_dict = torch.load('./model/nokkaew_model.pth')
    NokkaewGPT.load_state_dict(model_state_dict)

start = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
log_file = open("./log/"+start+".txt", "a")
log_file.write("Ran at " + start + "\n\n")

log_file.write("Reset: " + str(reset) + "\n\n")

log_file.write("Hyperparameter:\n")
log_file.write("batch_size: " + str(model.batch_size) + "\n")
log_file.write("block_size: " + str(model.block_size) + "\n")
log_file.write("n_embd: " + str(model.n_embd) + "\n")
log_file.write("n_head: " + str(model.n_head) + "\n")
log_file.write("n_layer: " + str(model.n_layer) + "\n")
log_file.write("dropout: " + str(model.dropout) + "\n")
log_file.write("vocab_size: " + str(model.vocab_size) + "\n\n")

log_file.write("Training steps: " + str(total_step) + "\n\n")

for steps in range(total_step):
    # sample a batch of data
    xb, yb = model.get_batch('train')

    # evaluate the loss
    logits, loss = NokkaewGPT(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Progress report every n steps
    if steps % 10 == 0 or steps == 0 or steps == total_step - 1:
        with torch.no_grad():
            val_xb, val_yb = model.get_batch('val')
            val_logits, val_loss = NokkaewGPT(val_xb, val_yb)

        if best_loss is None:
            best_loss = val_loss
        # update best loss
        if val_loss < best_loss:
            best_loss = val_loss
            # save the model
            torch.save(NokkaewGPT.state_dict(), './model/nokkaew_model.pth')
            log_file.write("The step below is saved!")
        log_file.write(str(steps) + '/' + str(total_step) + "\n")
        log_file.write(f'Step {steps}: Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}' + '\n\n')

# Evaluation on the test dataset
test_xb, test_yb = model.get_batch('test')
test_logits, test_loss = NokkaewGPT(test_xb, test_yb)

# Save test loss to the log file
log_file.write(f'Test Loss: {test_loss.item()}' + '\n')

log_file.close()
