import torch
import torch.nn as nn
from torch.nn import functional as F

class GPT(nn.Module):

    def __init__(self, config, n_tokens, ):
        super().__init__()
        self.n_tokens = n_tokens
        # to learn a embedding for each token
        # for simplicity we have set embedding_dim also equal to n_tokens, it is actually a hidden dim
        self.token_embedding_table = nn.Embedding(num_embeddings=n_tokens, embedding_dim=n_tokens) # so this a (n_tokens, n_tokens) matrix


    def forward(self, x, y=None):

        # x is the input whose size is (batch_size, block_size)
        # as the token_embedding_table helps to learn embedding for each token
        # each row of x is a chunk (block) of tokens, so along row we have the batch_size, each row has sequence of tokens
        # for each token, model learn an embedding of "embedding_dim" and those weights are stored in token_embedding_table
        # so the shape of logits become (batch_size, block_size, embedding_dim)
        # embedding_dim for our case is n_tokens

        logits = self.token_embedding_table(x) # shape of the logits will be (batch_size, block_size, n_tokens)


        #cross entropy at backend applies a softmax to logits, 
        # softmax is applied to each token in the n_tokens dimension 
        # (batch_size, block_size, any_idx) : this would be a 1-d vector to which softmax will be applied
        # After softmax, each token has a probability distribution over the vocab_size classes.
        if y is not None:
            #reshape because pytorch expects it like that
            batch_size, block_size, n_tokens = logits.shape
            #TODO: see why logits.view(batch_size, block_size, n_tokens) was producing higher loss 4.4 vs 2.5
            logits = logits.view(batch_size*block_size, n_tokens)
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)
            return logits, loss
        else: return logits

    def generate(self, x, max_new_tokens):
        # x is the input, size is (batch_size, block_size)
        for _ in range(max_new_tokens):
            logits = self.forward(x)
            #shape of the logits will be (batch_size, n_tokens, block_size) (we changed it in the forward function check)
            #now this logits will have tokens corresposding to whole x input, but we will be interested only for the last token along the time layer or chunk/block layer
            logits = logits[:, -1, :].view(-1, self.n_tokens) # now the shape of logits become (batch_size, n_tokens, 1) or (batch_size, n_tokens) after applying view
            #apply softmax along the n_tokens dim, this would help us arrive at the probabilities for classes
            probs = F.softmax(logits, dim=-1) # shape of the probs would still be (batch_size, n_tokens)
            # sample from the probabilitt distribution obtained
            x_next = torch.multinomial(probs, num_samples=1) # shape of this will be (batch_size, 1)
            # append sampled index to the running sequence
            x = torch.cat((x, x_next), dim=1) # shape will be (batch_size, block_size+1)
        return x



            
