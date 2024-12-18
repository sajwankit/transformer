from load_config import load_yaml_config
from dataset import TinyShakespeareDataset
from model import GPT

from torch.utils.data import DataLoader

class Trainer:

    def __init__(self, config, train_dataloader, val_dataloader, n_tokens):
        self.device = config["model"]["device"]
        self.model = GPT(config, n_tokens=n_tokens)


    def train_one_epoch(self):
        epoch_loss = 0
        for train_data in train_dataloader:
            logits, loss = self.model(x=train_data[0], y=train_data[1])
            epoch_loss += loss
        return epoch_loss/len(train_dataloader)

if __name__ == "__main__":
    config = load_yaml_config(r'/Users/ankitsajwan/tech/projects/transformer/config/config.yaml')
    train_dataset = TinyShakespeareDataset(config, mode="train")
    val_dataset = TinyShakespeareDataset(config, mode="val")

    train_dataloader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    n_tokens = len(train_dataset.char_to_token_map)
    trainer = Trainer(config, train_dataloader, val_dataloader, n_tokens)
    for train_data in train_dataloader:
        trainer.train_one_epoch()