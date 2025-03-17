import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
torch.manual_seed(42)

@dataclass
class LLMConfig:
    block_size = 512
    batch_size = 12
    n_layer = 12
    n_head = 12
    n_embd = 768
    hidden_dim = n_embd
    head_size = n_embd//n_head
    drop_out = 0.1
    vocab_size = 50274 # GPT官方的Tokenizer
    path = 'c'

class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.hidden_dim,config.head_size)
        self.value = nn.Linear(config.hidden_dim,config.head_size)
        self.query = nn.Linear(config.hidden_dim,config.head_size)
        
        self.register_buffer(
            "attention_mask",
            torch.tril(
                torch.ones(config.block_size,config.block_size)
                )
        )
        self.dropout = nn.Dropout(config.drop_out)    
    def forward(self,x):
        b_size, seq_len, hidden_dim = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        score = torch.matmul((q ,k.transpose(-2,-1)) / math.sqrt(hidden_dim))
        score = score.masked_fill(
            self.attention_mask[:seq_len,:seq_len]==0,
            float('-inf')
        )
        score = F.softmax(score,dim=-1)
        score = self.dropout(score)# dropout是随机丢弃一些权重
        out = score @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.hidden_dim,config.hidden_dim)
        self.dropout = nn.Dropout(config.drop_out)
        
    def forward(self,x):
        output = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim,config.hidden_dim*4),
            nn.GELU(),
            nn.Linear(config.hidden_dim*4,config.hidden_dim),
            nn.Dropout(config.drop_out)
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.Multihead = MultiHeadAttention(config)
        self.FFN = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
    def forward(self,x):
        x = x + self.Multihead(self.ln1(x))
        x = x + self.FFN(self.ln2(x))
        return x

class LLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embbedding = nn.Embedding(config.vocab_size,config.n_embd)
        self.p_ed = nn.Embedding(config.block_size,config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.lnfinal = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        
        # 权重绑定：输出层的权重是嵌入层权重的转置
        self.lm_head.weight = self.token_embbedding.weight
        
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0 ,std= 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std= 0.02)
            
    def forward(self,idx, tgt= None):
        # idx输入是Token id
        # tgt 是目标 token id
        # shape 一样
        batch ,seq_len = idx.size
        token_emd = self.token_embbedding(idx)
        pos_emb = self.p_ed(
            torch.arange(seq_len,device=idx.device)
        )
        # 经典题目： 为什么token_embbeding和position——embbeding是可以相加的
        x = token_emd + pos_emb
        x = self.blocks(x)
        x = self.lnfinal(x)
        x = self.lm_head(x)
        logits = F.softmax(x,dim=-1)       
        if tgt is None:
            loss= None
        else:
            batch_size, seq_len, vacab_size= logits.size
            logits = logits.view(batch_size*seq_len,vacab_size)
            tgt = tgt.view(batch_size*seq_len)
            loss = F.cross_entropy(logits,tgt)
        return logits,loss

class MyDataset(Dataset):
    def __init__(self,config):
        super().__init__()
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = config.block_size
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special = {"<|endoftext|>"}
        )[0]
        self.max_lines = 500
        self.encoded_data = []
        import json
        raw_data = []
        with open(config.path, 'r') as f:
            for i,line in enumerate(f):
                if i>= self.max_lines:
                    break
                try:
                    text = json.load(line.strip())['text']
                    raw_data.append(text)
                except Exception as e:
                    continue
        full_encode = []
        for text in raw_data:
            encode_text = self.enc.encode(text)
            full_encode.extend(encode_text+self.eos_token)
        
        for i in range(0,len(full_encode),self.block_size):
            chunk = full_encode[i:i+self.block_size+1]
            if len(chunk) < self.block_size+1:
                chunk = chunk + [self.eos_token]*(self.block_size +1 -len(chunk))
            self.encoded_data.append(chunk)
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1],dtype=torch.long)
        y = torch.tensor(chunk[-1:],dtype=torch.long)
        return x,y
    
model = LLM(LLMConfig())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params/1e6}'+'M')
    
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    