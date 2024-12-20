from load_config import load_yaml_config
from dataset import TinyShakespeareDataset
from model import GPT

import torch
from torch.utils.data import DataLoader
import time

torch.manual_seed(1337)

class Trainer:

    def __init__(self, config, train_dataloader, val_dataloader, n_tokens):
        self.device = config["model"]["device"]
        self.model = GPT(config, n_tokens=n_tokens).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["model"]["learning_rate"])
        self.batch_size = config["data"]["batch_size"]

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        for train_data in train_dataloader:
            if train_data[0].shape[0] != config["data"]["batch_size"]: continue
            # train_data = [x.to(self.device) for x in train_data]# move input data to mps or gpu
            logits, loss = self.model(x=train_data[0], y=train_data[1])

            self.optimizer.zero_grad(set_to_none=True)

            #back propagation step
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() # If you don’t call loss.item(), PyTorch will keep the computational graph for the loss.
            #This will lead to a memory leak because PyTorch will hold on to the entire computation graph for every batch.
        return epoch_loss/len(train_dataloader)
    
    def val_one_epoch(self):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for val_data in val_dataloader:
                if val_data[0].shape[0] != config["data"]["batch_size"]: continue
                # val_data = [x.to(self.device) for x in val_data]# move input data to mps or gpu
                logits, loss = self.model(x=val_data[0], y=val_data[1])
                epoch_loss += loss.item()
            return epoch_loss/len(val_dataloader)

    def train(self, n_epochs=10):
        for epoch in range(1, n_epochs+1):
            st = time.time()
            train_loss_epoch = self.train_one_epoch()
            val_loss_epoch = self.val_one_epoch()
            print(f"Epoch {epoch}: train_loss: {train_loss_epoch} | val_loss: {val_loss_epoch} | Time taken: {time.time()-st} secs")

if __name__ == "__main__":
    config = load_yaml_config(r'/Users/ankitsajwan/tech/projects/transformer/config/config.yaml')
    train_dataset = TinyShakespeareDataset(config, mode="train")
    val_dataset = TinyShakespeareDataset(config, mode="val")

    train_dataloader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    n_tokens = len(train_dataset.char_to_token_map)
    trainer = Trainer(config, train_dataloader, val_dataloader, n_tokens)

    def generated_text(x=torch.zeros((32,1), dtype=torch.long), max_new_tokens=500):
        generated_tokens = trainer.model.generate(x, max_new_tokens).view(-1).tolist()
        return train_dataset.decoder(generated_tokens)
    print("*********before traininG***********)")
    # print(generated_text())
    trainer.train(config["model"]["epochs"])
    print("************after training****************")
    # print(generated_text())  
    
    
    