from load_config import load_yaml_config

import numpy as np
import torch 
import tiktoken
from torch.utils.data import Dataset, DataLoader

class TinyShakespeareDataset(Dataset):

    def __init__(self, config, mode="train"):
        with open(config["data"]["data_path"]) as file:
            self.data = file.read()
        split_end = int(config["data"]["train_val_split"]*len(self.data))
        self.mode = mode
        if mode == "train":
            self.data_tokens = self.get_encoding(self.data[:split_end])
        else:
            self.data_tokens = self.get_encoding(self.data[split_end: ])
        self.block_size = config["data"]["block_size"]
        # self.encoding = tiktoken.get_encoding("gpt2")
        # self.encoding.encode(self.data)

    def get_encoding(self, text):
        chars_in_text = set(text)
        char_to_int_map = {ch: i for i, ch in enumerate(text)}
        tokens = [char_to_int_map[ch] for ch in text]
        return tokens

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data_tokens[idx: idx+self.block_size].astype(np.int64))
        if self.mode == "train":   
            y = torch.from_numy(self.data_tokens[idx+1: idx+self.block_size+1].astype(np.int64)) 
            return x, y
        else:   return x
    
    def __len__(self):
        """
        returns max length which is possible
        """
        return len(self.data_tokens) - self.block_size
    
if __name__=="__main__":
    config = load_yaml_config(r'/Users/ankitsajwan/tech/projects/transformer/config/config.yaml')
    train_dataset = TinyShakespeareDataset(config, mode="train")
    val_dataset = TinyShakespeareDataset(config, mode="val")