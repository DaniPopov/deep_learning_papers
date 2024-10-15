import torch
import torch.nn as nn
import math

class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear layers for Q, K, V 
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Final linear layer after concatenation
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        # Q - query, K - key, V - value
        # Q -  ask "what information do I need?"
        # K -  determine "what information do I provide?"
        # V -  supply "the information that will be used."
        batch_size = Q.shape[0]

        # Use the linear layers to project Q, K, V
        Q_proj = self.q_linear(Q)  # [batch_size, seq_len, d_model]
        K_proj = self.k_linear(K)  # [batch_size, seq_len, d_model]
        V_proj = self.v_linear(V)  # [batch_size, seq_len, d_model]

        # Reshape form [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, head_dim]
        Q_proj = Q_proj.reshape(Q_proj.shape[0], Q_proj.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        K_proj = K_proj.reshape(K_proj.shape[0], K_proj.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        V_proj = V_proj.reshape(V_proj.shape[0], V_proj.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention for each head using Scaled Dot-Product Attention (vectorized implementation)
        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attention_output = torch.matmul(attention_weights, V_proj)  # [batch_size, num_heads, seq_len, head_dim]

        # Concatenate the heads' output: [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        concatenated_output = attention_output.reshape(batch_size, -1, self.num_heads * self.head_dim)  # [batch_size, seq_len, d_model]

        # Apply the final linear projection
        output = self.out_proj(concatenated_output)  # [batch_size, seq_len, d_model]

        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.self_attn = Multi_Head_Attention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, Q, K, V):
        # Multi-head attention with residual connection and normalization
        Q = self.norm1(Q + self.self_attn(Q, K, V)[0])
        
        # Feed-forward network with residual connection and normalization
        Q = self.norm2(Q + self.feed_forward(Q))
        
        return Q

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.self_attn = Multi_Head_Attention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = Multi_Head_Attention(d_model, num_heads)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # Self-attention with causal mask
        attn_output = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + attn_output)
        
        # Cross-attention
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, mask=src_mask)
        x = self.norm2(x + cross_attn_output)
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x

# Add this function to generate the causal mask
def generate_square_subsequent_mask(sz: int):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class Encoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
    
    def forward(self, Q, K, V):
        for layer in self.layers:
            Q = layer(Q, K, V) 
        return Q
    
class Decoder(nn.Module):
    def __init__(self, d_model=512, num_layers=6, num_heads=8, d_ff=2048):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

def main():
    vocab_size = 10000  # Size of your vocabulary
    d_model = 512  # Dimension of the model
    input_tensor = torch.randint(0, vocab_size, (2, 10))
    embedding = Embedding(vocab_size, d_model)
    output = embedding(input_tensor)
    

if __name__ == "__main__":
    main()
