import argparse
from dataclasses import dataclass
import json
import math
import os
import pandas as pd
from pathlib import Path
import re
from sys import exit
from time import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

def clean_recipe_names(recipe_df, column_name = 'recipe_name', remove_parantheticals = True):
    '''
    Cleans the recipe names column of a dataframe to contain only English alpha-numeric and space characters
    TODO: Create a data class, and migrate this function into it.
        
    Parameters
    ----------
    recipe_metadata_df : Pandas DataFarme
        A DataFrame which contains a recipe name column
    column_name: str, optional
        The name of the column which contains recipe names
    remove_parantheticals : bool, optional
        If true, removes all text between parantheses characters "(" and ")" before dropping other data
        
    Returns
    -------
    recipe_metadata_df : Pandas DataFarme
        The DataFrame of cleaned names

    '''
    recipe_names = recipe_df[column_name].astype(str).tolist()
    
    if remove_parantheticals:
        recipe_names = [re.sub(r'\((.*?)\)', '', name) for name in recipe_names]
        recipe_names = [re.sub(r'[()]', '', name) for name in recipe_names]
    
    recipe_names = [re.sub(r'[^a-zA-Z0-9 ]', '', name) for name in recipe_names]
    recipe_names = [name.strip() for name in recipe_names]
    
    recipe_df[column_name] = recipe_names
    
    return recipe_df

def split_df_into_train_test(df, train_proportion = 0.9): 
    '''
    Creates two randomly sampled and specificaly proportioned DataFrames from an input.
    TODO: Create a data class, and migrate this function into it.

    Parameters
    ----------
    df : Pandas DataFrame
        The DataFrame to be split into training and testing sets
    train_proportion : float, optional
        The proportion of data to be used for testing. The default is 0.85.

    Returns
    -------
    training_df : Pandas DataFrame
        A randomly selected number of rows from the input data frame.
    testing_df : Pandas DataFrame
        A randomly selected number of rows from the input data frame.

    '''
    df_size = df.index.size
    train_size = int(train_proportion * df_size)
        
    random_indicies = torch.randperm(df_size).tolist()
    train_indices = random_indicies[:train_size]
    test_indices = random_indicies[train_size:]
    
    training_df = df[df.index.isin(train_indices)]
    testing_df = df[df.index.isin(test_indices)]
    
    return training_df, testing_df

def load_bjcp_styles(json_file_path):
    '''
    Loads the 2021 BJCP JSON styleguide obtained from https://github.com/beerjson/bjcp-json
    TODO: Create a data class, and migrate this function into it.
        
    Parameters
    ----------
    file_path : str
        The file path to the BJCP style guide JSON to be read.
        
    Returns
    -------
    styles_dict : dict
        A dictionary of BJCP styles indexed by the style id (ex. '1A', '16D', 'X1')

    '''

    with open(json_file_path, encoding = 'cp850') as bjcp_file:
        styles_list = json.load(bjcp_file)
    
    styles_list = styles_list['beerjson']['styles']
    styles_index = [i['style_id'] for i in styles_list]
    
    styles_dict = {k:v for (k, v) in zip(styles_index, styles_list)}
    
    return styles_dict
    
# The functions below are taken directly from Andrej Karpathy's MakeMore, accessed at https://github.com/karpathy/makemore/ on 12/12/2023
# (Thank you Andrej!)
@dataclass
class ModelConfig:
    block_size: int = None # length of the input sequences of integers
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4


# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to('cuda') for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y    
    
class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch
    
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = nn.functional.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def print_samples(num=10):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to('cuda')
   # top_k = args.top_k if args.top_k != -1 else None
    top_k = None
    steps = training_dataset.get_output_length() - 1 #because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cuda')
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = training_dataset.decode(row)
        # separately track samples that we have and have not seen before
        if training_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif testing_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print('-'*80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print('-'*80)

if __name__=="__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate Recipe Names")
    parser.add_argument('--input-dir', '-i', type=str, default='Data', help="Path to the input directory.")
    parser.add_argument('--output-dir', '-o', type=str, default='Models/NameGenerator', help="Path to the output directory.")
    # Argparse inverts the values of False and True when stored using "action =". Not sure why.
    parser.add_argument('--resume', action='store_false', help="When True, the last saved model will be used.")
    parser.add_argument('--sample-only', action='store_true', help="When True, the script will only sample from an existing model, not create a new one.")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="The number of sub-processors to allow PyTorch to use in training/testing. More workers means more memory usage.")
    parser.add_argument('--max-steps', type=int, default=5000, help="How many steps of training/testing optimization should be run. A value of -1 will result in infinite steps.")
    parser.add_argument('--device', type=str, default='cuda', help="The device to use for computing. Options are: 'cpu,' 'cuda', 'cuda:2', or 'mps'.")
    parser.add_argument('--seed', type=int, default=76834, help="Random seed.")
    parser.add_argument('--top-k', type=int, default=-1, help="Used to return the largest K values during sampling. A value of -1 will result in no top-K sampling.")
    parser.add_argument('--type', type=str, default='transformer', help="What type of model should be used. 'transformer' is currently the only option")
    parser.add_argument('--n-layer', type=int, default=4, help="The number of layers (blocks) in the model.")
    parser.add_argument('--n-head', type=int, default=4, help="The number of heads for multi-head attention (Transformer model).")
    parser.add_argument('--n-embd', type=int, default=64, help="The number of feature embeddings used in the model.")
    parser.add_argument('--n-embd2', type=int, default=64, help="The number of secondary feature embeddings used in the model (Unused by Transformer model).")
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="The number of batches to be simultaneously run during training/testing optimization.")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="The optimizaer learning rate.")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="The optimizaer weight decay.")
    args = parser.parse_args()
    
    # Set up system
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    output_path_base = args.output_dir
    Path(output_path_base).mkdir(parents=True, exist_ok=True)
    
    writer_output_path = f'{output_path_base}/Writer'
    Path(writer_output_path).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(writer_output_path)
    
    input_path_basse = args.input_dir
    
    recipe_metadata = pd.read_csv(f'{input_path_basse}/brewersfriend/metadata.csv')
    
    # Clean recipe names
    recipe_metadata = clean_recipe_names(recipe_metadata)
    recipe_metadata = recipe_metadata.drop_duplicates(subset = ['recipe_name'])
    
    # Clean recipe metadat to only include those tagged with proper BJCP styles
    # TODO: There may be some error here if beers were entered with earlier BJCP styles. Conversion from 2015 and 2017 guidelines to 2019 guidelines might be necessary.
    recipe_metadata = recipe_metadata[recipe_metadata['style_guide'] == 'BJCP']
    recipe_metadata['recipe_bjcp_style_id'] = recipe_metadata['style_category_number'].astype(str) + recipe_metadata['style_letter']
    
    bjcp_2019_styles = load_bjcp_styles('Data/styleguides/bjcp_styleguide-2021.json')
    recipe_metadata = recipe_metadata[recipe_metadata['recipe_bjcp_style_id'].isin(list(bjcp_2019_styles.keys()))] 
    
    # Get necessary model variables from recipe name data, and creating training/testing datasets
    largest_recipe_len = recipe_metadata['recipe_name'].map(len).max()
    recipe_name_chars = sorted(list(set(''.join(recipe_metadata['recipe_name']))))

    training_metadata, testing_metadata = split_df_into_train_test(recipe_metadata)
    training_dataset = CharDataset(training_metadata['recipe_name'].tolist(), recipe_name_chars, largest_recipe_len)
    testing_dataset = CharDataset(testing_metadata['recipe_name'].tolist(), recipe_name_chars, largest_recipe_len)

    vocab_size = training_dataset.get_vocab_size()
    block_size = training_dataset.get_output_length()

    this_model_config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)

    model = Transformer(this_model_config) # If more models are added, they should be selected here
    model.to(args.device)
    model_file_name = f'{output_path_base}/gen_recipe_name_{args.type}.pt'
    
    if args.resume or args.sample_only:
        if os.path.isfile(model_file_name):    
            print(f"Resuming from existing model at {model_file_name}.")
            model.load_state_dict(torch.load(model_file_name))
        else:
           print(f"Warning: Could not find existing model {model_file_name}. Starting with new model.")
           
    if args.sample_only:
        print_samples(num=50)
        exit()
    
    print('Creating optimizer')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

    print('Creating batch loader')
    batch_loader = InfiniteDataLoader(training_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    print('Starting training loop')
    best_loss = None
    step = 0
    while True:
        start_time = time()
        
        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # feed into the model
        logits, loss = model(X, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        
        current_time = time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(current_time-start_time)*1000:.2f}ms")

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, training_dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, testing_dataset,  batch_size=100, max_batches=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                print(f"test loss {test_loss} is the best so far, saving model to {model_file_name}")
                torch.save(model.state_dict(), model_file_name)
                best_loss = test_loss

        # sample from the model
        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1
        # termination conditions
        if args.max_steps > 0 and step > args.max_steps:
            break