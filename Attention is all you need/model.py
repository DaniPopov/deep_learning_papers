import torch
import torch.nn as nn
import math

class Dot_Product_Attention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V):
        d_k = K.size(-1)

        # Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        attention = torch.matmul(attention_weights, V)
        return attention

class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear layers for Q, K, V projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Final linear layer after concatenation
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
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
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.attention = Multi_Head_Attention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, Q, K, V):
        # Multi-head attention with residual connection and normalization
        attn_output = self.attention(Q, K, V)
        Q = self.norm1(Q + attn_output)
        
        # Feed-forward network with residual connection and normalization
        ffn_output = self.ffn(Q)
        output = self.norm2(Q + ffn_output)
        
        return output


class Encoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
    
    def forward(self, Q, K, V):
        for layer in self.layers:
            Q = layer(Q, K, V) 
        return Q 

def main():
    test = True
    if test:
        Q = torch.randn([4,10,16])
        K = torch.rand([4,10,16])
        V = torch.rand([4,10,16])
        dot_product = Dot_Product_Attention()
        attention = dot_product(Q, K, V)
        print(attention.shape)

if __name__ == "__main__":
    main()

    