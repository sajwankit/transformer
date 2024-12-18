import torch
import torch.nn as nn
from torch.nn import functional as F

class GPT(nn.Module):

    def __init__(self, config, n_tokens, ):
        super().__init__()

        # to learn a embedding for each token
        # for simplicity we have set embedding_dim also equal to n_tokens, it is actually a hidden dim
        self.token_embedding_table = nn.Embedding(num_embeddings=n_tokens, embedding_dim=n_tokens) # so this a (n_tokens, n_tokens) matrix


    def forward(self, x, y):

        # x is the input whose size is (batch_size, block_size)
        # as the token_embedding_table helps to learn embedding for each token
        # each row of x is a chunk (block) of tokens, so along row we have the batch_size, each row has sequence of tokens
        # for each token, model learn an embedding of "embedding_dim" and those weights are stored in token_embedding_table
        # so the shape of logits become (batch_size, block_size, embedding_dim)
        # embedding_dim for our case is n_tokens

        logits = self.token_embedding_table(x) # shape of the logits will be (batch_size, block_size, n_tokens)


        #reshape because pytorch expects it like that
        batch_size, block_size, n_tokens = logits.shape
        logits = logits.view(batch_size, n_tokens, block_size)


        #cross entropy at backend applies a softmax to logits, 
        # softmax is applied to each token in the n_tokens dimension 
        # (batch_size, block_size, any_idx) : this would be a 1-d vector to which softmax will be applied
        # After softmax, each token has a probability distribution over the vocab_size classes.
        loss = F.cross_entropy(logits, y)
        return logits, loss
