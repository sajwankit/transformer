import torch
import torch.nn as nn
from torch.nn import functional as F



class SimpleAverageSelfAttention(nn.Module):

    """
    Take just simple average of past token to make current token aware about past
    Establish communication between the tokens
    so for a current token, some weighted aggregation of past tokens can be taken so that the current token becomes aware of the context of past tokens.
    NOTE: also past token context passed as we are predicting the future tokens, so the model should not see the future tokens

    To establish communication between tokens so that the current token can be aware of the context of past tokens (without seeing future tokens), 
    we use the self-attention mechanism with causal masking. 

    Initially, one might compute a simple average of past tokens to represent a word, which doesn’t account for the varying importance of different tokens.
    """
    def __init__(self, config):
        super().__init__()
        # This approach allows for efficient matrix-based computation. 
        # When a matrix  M  is multiplied with a lower triangular mask, each row in the result contains the average of the accessible elements up to that point:
            #•	The last row includes the average of all elements in the row.
            #•	The second-last row averages all its elements except the last.
            #•	This pattern continues, with the first row averaging only its first element.
        # Thus, each row of the final matrix reflects the context for the next token in the sequence, 
        # ensuring that each token has access only to past and current tokens. 
        # This approach enables faster computation of self-attention using matrix operations instead of iterative loops.
        # use register_buffer to add a non-trainable constant which do not change with training
        chunk_size = config["data"]["chunk_size"]
        batch_size = config["data"]["batch_size"]
        self.register_buffer('causal_mask', torch.tril(torch.ones(chunk_size, chunk_size)))
        self.register_buffer('attention_weights', torch.zeros((batch_size, chunk_size, chunk_size)))
    
    def forward(self, x):
        # attention weights size is (batch_size, chunk_size, chunk_size)
        # x which is the input, is basically the logits which we obtain with token_embedding_table, size is (batch_size, chunk_size, one_token_emb_size)
        attention_weights = self.attention_weights.masked_fill(self.causal_mask==0, value=-float('inf')) # the mask would be broadcasted to all of the rows of the batch
        attention_weights = F.softmax(attention_weights, dim=-1)

        # (batch_size, chunk_size, chunk_size) @ (batch_size, chunk_size, one_token_emb_size) --> (batch_size, chunk_size, one_token_emb_size)
        # so let's say the first row of the batch --> (1, chunk_size, one_token_emb_size)
        # attention layer outputs contextualised embeddings, what that means is that each token embedding in it (along the one_token_emb_size) is the output token after considering all the past tokens

        """
        To understand the role of contextualization, think of how the word “it” changes meaning depending on the sentence:
            •	“The cat chased the mouse, and it escaped.” (Here, “it” refers to the mouse.)
            •	“I got a new phone, and it has a great camera.” (Here, “it” refers to the phone.)

        The model should understand that the meaning of “it” is context-dependent.
            •	The base embedding of “it” (from nn.Embedding) is the same in both cases.
            •	But the contextualized embedding for “it” will be different because it aggregates information from the past tokens (“cat”, “mouse”, “phone”, etc.) in each sentence.
            •	The attention layer is what makes this possible.
        """
        attention_output = attention_weights @ x
        return attention_output
        
        

class GPT(nn.Module):

    def __init__(self, config, n_tokens, ):
        super().__init__()
        self.chunk_size = config["data"]["chunk_size"]
        self.n_tokens = n_tokens
        # we are learning an embedding for each token
        self.one_token_emb_size = config["model"]["token_emb_size"]
        self.token_embedding_table = nn.Embedding(num_embeddings=n_tokens, embedding_dim=self.one_token_emb_size) # so this a (n_tokens, one_token_emb_size) matrix
        
        # token_embedding_table captures just the identity of the token, position embedding will capture the position of tokens in a chunk
        #TODO read how
        self.position_embedding_table = nn.Embedding(self.chunk_size, self.one_token_emb_size)

        self.attention_layer = SimpleAverageSelfAttention(config)

        self.linear_layer = nn.Linear(self.one_token_emb_size, self.n_tokens) # but to compare to the final output, we would need n_tokens dimension



    def forward(self, x, y=None):

        # x is the input whose size is (batch_size, chunk_size)
        # as the token_embedding_table helps to learn embedding for each token
        # each row of x is a chunk (block) of tokens, so along row we have the batch_size, each row has sequence of tokens
        # for each token, model learn an embedding of "embedding_dim" and those weights are stored in token_embedding_table
        # so the shape of logits become (batch_size, chunk_size, embedding_dim)
        # embedding_dim for our case is n_tokens

        token_embeddings = self.token_embedding_table(x) # shape of the token_embeddings will be (batch_size, chunk_size, one_token_emb_size)


        position_embeddings = self.position_embedding_table(torch.arange(self.chunk_size)) # shape is (chunk_size, one_token_emb_size)

        token_embeddings = token_embeddings + position_embeddings # shape is (batch_size, chunk_size, one_token_emb_size)
    
        contextualised_embeddings = self.attention_layer(token_embeddings)

        logits = self.linear_layer(contextualised_embeddings)# shape of the logits will be (batch_size, chunk_size, n_tokens)

        #cross entropy at backend applies a softmax to logits, 
        # softmax is applied to each token in the n_tokens dimension 
        # (batch_size, chunk_size, any_idx) : this would be a 1-d vector to which softmax will be applied
        # After softmax, each token has a probability distribution over the vocab_size classes.
        if y is not None:
            #reshape because pytorch expects it like that
            batch_size, chunk_size, n_tokens = logits.shape
            #TODO: see why logits.view(batch_size, chunk_size, one_token_emb_size) was producing higher loss 4.4 vs 2.5
            logits = logits.view(batch_size*chunk_size, n_tokens)
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)
            return logits, loss
        else: return logits

    def generate(self, x, max_new_tokens):
        # x is the input, size is (batch_size, chunk_size)
        for _ in range(max_new_tokens):
            x_cropped = x[:, -self.chunk_size:]
            logits = self.forward(x_cropped)
            #shape of the logits will be (batch_size, n_tokens, chunk_size) (we changed it in the forward function check)
            #now this logits will have tokens corresposding to whole x input, but we will be interested only for the last token along the time layer or chunk/block layer
            logits = logits[:, -1, :].view(-1, self.n_tokens) # now the shape of logits become (batch_size, n_tokens, 1) or (batch_size, n_tokens) after applying view
            #apply softmax along the n_tokens dim, this would help us arrive at the probabilities for classes
            probs = F.softmax(logits, dim=-1) # shape of the probs would still be (batch_size, n_tokens)
            # sample from the probabilitt distribution obtained
            x_next = torch.multinomial(probs, num_samples=1) # shape of this will be (batch_size, 1)
            # append sampled index to the running sequence
            x = torch.cat((x, x_next), dim=1) # shape will be (batch_size, chunk_size+1)
        return x



            
