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
        # x which is the input, is basically the logits which we obtain with token_embedding_table, size is (batch_size, chunk_size, token_emb_size)
        attention_weights = self.attention_weights.masked_fill(self.causal_mask==0, value=-float('inf')) # the mask would be broadcasted to all of the rows of the batch
        attention_weights = F.softmax(attention_weights, dim=-1)

        # (batch_size, chunk_size, chunk_size) @ (batch_size, chunk_size, token_emb_size) --> (batch_size, chunk_size, token_emb_size)
        # so let's say the first row of the batch --> (1, chunk_size, token_emb_size)
        # attention layer outputs contextualised embeddings, what that means is that each token embedding in it (along the token_emb_size) is the output token after considering all the past tokens

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
        

class SingleSelfAttentionHead(nn.Module):
    """ one head of self-attention"""

    def __init__(self, chunk_size, token_emb_dim, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(token_emb_dim, head_size, bias=False)
        self.query = nn.Linear(token_emb_dim, head_size, bias=False)
        self.value = nn.Linear(token_emb_dim, head_size, bias=False)
        self.register_buffer("causal_mask", torch.tril(torch.ones(chunk_size, chunk_size)))
    
    def forward(self, x):
        batch_size, chunk_size, token_emb_dim = x.shape 
        key_vector = self.key(x) # size will (batch_size, chunk_size, head_size)
        query_vector = self.query(x) # size will (batch_size, chunk_size, head_size)

        # compute attention weights 
        attention_weights = query_vector @ key_vector.transpose(-2,-1) # size will be (batch_size, chunk_size, chunk_size)
        #TODO: add comments why we do this
        attention_weights = attention_weights * (self.head_size**-0.5)

        attention_weights = attention_weights.masked_fill(self.causal_mask==0, -float('inf'))
        attention_weights = F.softmax(attention_weights, dim=-1) 

        value_vector = self.value(x) # size will (batch_size, chunk_size, head_size)
        
        contextualised_embeddings = attention_weights @ value_vector # shape will be (batch_size, chunk_size, head_size)

        return contextualised_embeddings

class MultiHeadSelfAttention(nn.Module):
    """multiple heads of self-attention in PARALLEL"""

    def __init__(self, n_heads, chunk_size, token_emb_dim):
        super().__init__()
        head_size = token_emb_dim // n_heads
        single_attention_head = SingleSelfAttentionHead(chunk_size, token_emb_dim, head_size)
        self.heads = nn.ModuleList([single_attention_head for _ in range(n_heads)])

        #TODO not clear why this is need for residual pathway, confirm below thinking
        """
        Without the Projection Layer:

            If there were no projection layer, the output of the residual connection would simply be:


            modeloutput = x + out1


            where:
                •	x is the input to the residual block.
                •	\text{out1} is the output from the sub-layer (e.g., multi-head attention or feed-forward network).

            In this case,  \text{out1}  is directly added to x without any learnable transformation.

        With the Projection Layer:

            When you add a projection layer (like self.proj), the output of the residual connection becomes:


            modeloutput = x + (W * out1+ b)
            By adding this transformation, the model can:
                1.	Learn a more refined transformation of \text{out1} before adding it to x.
                2.	Align dimensions if x and \text{out1} have different sizes.
                3.	Adjust the scale or representation of \text{out1} to make the addition x + f{\prime}(\text{out1}) more meaningful.
        
        """
        self.projection = nn.Linear(token_emb_dim, token_emb_dim)

    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, emb_dim_size):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim_size, 4 * emb_dim_size), # the 4 times in the inner layer added later per the transformer paper
            nn.ReLU(),
            nn.Linear(4 * emb_dim_size, emb_dim_size) # this layer is for residual projection
        )

    def forward(self,x):
        return self.feed_forward(x)
    
class TransformerBlock(nn.Module):
    """transformer block: the self attention and feed forward are combined as a block, as these would be repeated in the model
    so a transformer block conists of the communication (self attention) & computation (feed forward)
    """

    def __init__(self, config):
        super().__init__()
        self.n_self_attention_heads = config["model"]["n_self_attention_heads"]
        self.head_size = config["model"]["token_emb_size"] // self.n_self_attention_heads
        self.multi_head_self_attn = MultiHeadSelfAttention(
            n_heads=self.n_self_attention_heads,
            chunk_size=config["data"]["chunk_size"],
            token_emb_dim=config["model"]["token_emb_size"]
        )
        self.feed_forward = FeedForward(emb_dim_size=config["model"]["token_emb_size"])
    
    def forward(self, x):
        # now you would notice that that our model has become quite deeeeeep, with multiple transformer blocks, and within each block, multiple heads
        # use RESIDUAL CONNECTIONS TO HELP MODEL CONVERGE BETTER BY SOLVING VANISHING GRADIENT PROBLEM and other optimization issue 
        # Deep networks without residual connections can suffer from degradation problems, 
        # where adding more layers leads to worse performance (not due to overfitting, but because optimization becomes harder).
        # Residual connections allow very deep networks to converge effectively, making them scalable to hundreds or thousands of layers.
        x = x + self.multi_head_self_attn(x)
        x = x + self.feed_forward(x)
        return x


class GPT(nn.Module):

    def __init__(self, config, n_tokens, ):
        super().__init__()
        self.chunk_size = config["data"]["chunk_size"]
        self.n_tokens = n_tokens
        # we are learning an embedding for each token
        self.token_emb_size = config["model"]["token_emb_size"]
        self.token_embedding_table = nn.Embedding(num_embeddings=n_tokens, embedding_dim=self.token_emb_size) # so this a (n_tokens, token_emb_size) matrix
        
        # token_embedding_table captures just the identity of the token, position embedding will capture the position of tokens in a chunk
        #TODO read how
        self.position_embedding_table = nn.Embedding(self.chunk_size, self.token_emb_size)

        # self.attention_layer = SimpleAverageSelfAttention(config)
        
        # self.attention_layer = SingleSelfAttentionHead(chunk_size=self.chunk_size, token_emb_dim=self.token_emb_size, head_size=config["model"]["self_attention_head_size"])

        self.n_self_attention_heads = config["model"]["n_self_attention_heads"]
        self.head_size = self.token_emb_size // self.n_self_attention_heads
        self.attention_layer = MultiHeadSelfAttention(
            n_heads=self.n_self_attention_heads,
            chunk_size=self.chunk_size, 
            token_emb_dim=self.token_emb_size
        )

        self.feed_forward = FeedForward(emb_dim_size=self.n_self_attention_heads*self.head_size) # for understanding I've written it as self.n_self_attention_heads*self.head_size which is equal to token_emb_size

        self.transformer_blocks = nn.Sequential(
            TransformerBlock(config),
            TransformerBlock(config),
            TransformerBlock(config),
            TransformerBlock(config),
        ) # using 4 transformer blocks

        self.linear_layer = nn.Linear(self.n_self_attention_heads*self.head_size, self.n_tokens) # but to compare to the final output, we would need n_tokens dimension



    def forward(self, x, y=None):

        # x is the input whose size is (batch_size, chunk_size)
        # as the token_embedding_table helps to learn embedding for each token
        # each row of x is a chunk (block) of tokens, so along row we have the batch_size, each row has sequence of tokens
        # for each token, model learn an embedding of "embedding_dim" and those weights are stored in token_embedding_table
        # so the shape of logits become (batch_size, chunk_size, embedding_dim)
        # embedding_dim for our case is n_tokens

        token_embeddings = self.token_embedding_table(x) # shape of the token_embeddings will be (batch_size, chunk_size, token_emb_size)


        position_embeddings = self.position_embedding_table(torch.arange(self.chunk_size)) # shape is (chunk_size, token_emb_size)

        token_embeddings = token_embeddings + position_embeddings # shape is (batch_size, chunk_size, token_emb_size)
    
        # contextualised_embeddings = self.attention_layer(token_embeddings)

        contextualised_embeddings = self.transformer_blocks(token_embeddings)


        ff_output = self.feed_forward(contextualised_embeddings)

        logits = self.linear_layer(ff_output)# shape of the logits will be (batch_size, chunk_size, n_tokens)

        #cross entropy at backend applies a softmax to logits, 
        # softmax is applied to each token in the n_tokens dimension 
        # (batch_size, chunk_size, any_idx) : this would be a 1-d vector to which softmax will be applied
        # After softmax, each token has a probability distribution over the vocab_size classes.
        if y is not None:
            #reshape because pytorch expects it like that
            batch_size, chunk_size, n_tokens = logits.shape
            #TODO: see why logits.view(batch_size, chunk_size, token_emb_size) was producing higher loss 4.4 vs 2.5
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



            
