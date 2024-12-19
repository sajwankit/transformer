from load_config import load_yaml_config

import numpy as np
import torch 
import tiktoken
from torch.utils.data import Dataset

class TinyShakespeareDataset(Dataset):

    def __init__(self, config, mode="train"):
        self.device = config["model"]["device"]
        with open(config["data"]["data_path"]) as file:
            self.data = file.read()
        
        chars_in_data = set(self.data)
        self.char_to_token_map = {ch: i for i, ch in enumerate(chars_in_data)}
        self.token_to_char_map = {i: ch for ch, i in self.char_to_token_map.items()}

        split_end = int(config["data"]["train_val_split"]*len(self.data))
        self.mode = mode
        if mode == "train":
            self.data_tokens = self.encoder(self.data[:split_end])
        else:
            self.data_tokens = self.encoder(self.data[split_end: ])
        self.block_size = config["data"]["block_size"]
        
        # self.encoding = tiktoken.get_encoding("gpt2")
        # self.encoding.encode(self.data)

    def encoder(self, text):
        tokens = np.array([self.char_to_token_map[ch] for ch in text])
        return tokens
    
    def decoder(self, tokens):
        text = "".join([self.token_to_char_map[token] for token in tokens])
        return text

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data_tokens[idx: idx+self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data_tokens[idx+1: idx+self.block_size+1].astype(np.int64)) 
        return x.to(self.device), y.to(self.device)
    
    def __len__(self):
        """
        returns max length which is possible
        """
        return len(self.data_tokens) - self.block_size
    
if __name__=="__main__":
    from torch.utils.data import DataLoader
    config = load_yaml_config(r'/Users/ankitsajwan/tech/projects/transformer/config/config.yaml')
    train_dataset = TinyShakespeareDataset(config, mode="train")
    val_dataset = TinyShakespeareDataset(config, mode="val")
    
    #total number of tokens in the whole input text
    N_TOKENS = len(set(train_dataset.data_tokens) + set(val_dataset.data_tokens))

    train_dataloader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    for train_data in train_dataloader:
        print(train_data[0])
        break