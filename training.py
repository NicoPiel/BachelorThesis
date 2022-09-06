#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Imports

import os
import re
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# PyTorch Lightning
import pytorch_lightning as pl
import seaborn as sns

# PyTorch
import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import torchtext as tt
from torchtext.vocab import build_vocab_from_iterator

import torchmetrics.functional as metrics

from tqdm.notebook import tqdm
from IPython.display import set_matplotlib_formats
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from nltk.tokenize import RegexpTokenizer
import wandb

DEVICE = torch.device("cpu")

# Import GPU-related things
if torch.cuda.is_available():
    # import cupy as np
    # import cudf as pd

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    DEVICE = torch.device("cuda:0")
# else:

# Plotting
plt.set_cmap("cividis")
#%matplotlib inline
set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/")

# Setting the seed
pl.seed_everything(42)

print('CUDA:', torch.cuda.is_available())
print("Device:", DEVICE)


# In[12]:


files = [
    'data.csv',
    'edrug3d.sdf',
    'qm9-1.sdf',
    'qm9-2.sdf',
    'qm9-3.sdf',
    'qm9-4.sdf',
    'qm9-5.sdf',
    'qm9-6.sdf',
    'qm9-7.sdf',
    'qm9-8.sdf'
]


def check_missing_files():
    """Checks for missing files. Returns true, if all files are present."""
    for file in files:
        if not os.path.exists('./data/' + file):
            return False

    return True


if not check_missing_files():
    get_ipython().system('wget -nc -O data.zip "https://hochschulebonnrheinsieg-my.sharepoint.com/:u:/g/personal/nico_piel_365h-brs_de1/ESuGOTn_IflEk7I5HkOFpbwBZKeOk9Qf2nL5JEcq2om6_Q?e=sHYsTk&download=1"')
    get_ipython().system('unzip -u data.zip')
    get_ipython().system('rm data.zip')


# In[13]:


def in_ipython():
    try:
        return __IPYTHON__
    except NameError:
        return False


# In[14]:


class CustomDataset(data.Dataset):
    def __init__(self, path):
        super().__init__()
        PAD_TOKEN = '<PAD>'
        BOS_TOKEN = '<BOS>'
        EOS_TOKEN = '<EOS>'

        print('Reading csv..')
        self.data = pd.read_csv(path)

        # SMILES regex by Schwaller et. al.
        self.smiles_regex = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
        self.at_regex = r"\w+|\w+"

        self.smiles_tokenizer = RegexpTokenizer(self.smiles_regex)
        self.at_tokenizer = RegexpTokenizer(self.at_regex)

        # Build vocabs
        print('Building vocabs..')
        self.smiles_vocab = build_vocab_from_iterator(tqdm([self.smiles_tokenizer.tokenize(feature[0]) for feature in self.data.iloc]))
        self.at_vocab = build_vocab_from_iterator(tqdm([self.at_tokenizer.tokenize(feature[1]) for feature in self.data.iloc]))

        self.smiles_vocab.append_token(PAD_TOKEN)
        self.smiles_vocab.append_token(BOS_TOKEN)
        self.smiles_vocab.append_token(EOS_TOKEN)
        self.at_vocab.append_token(PAD_TOKEN)
        self.at_vocab.append_token(BOS_TOKEN)
        self.at_vocab.append_token(EOS_TOKEN)

        self.smiles_vocab_len = len(self.smiles_vocab)
        self.at_vocab_len = len(self.at_vocab)

        outer_smiles_vocab_len = self.smiles_vocab_len
        outer_at_vocab_len = self.at_vocab_len

        self.max_length = np.max(np.array([len(features[0]) for features in self.data.iloc]))

        seq_tensors = []

        print('Inserting special tokens..')
        for series in tqdm(self.data.iloc):
            smiles_tokens = self.smiles_tokenizer.tokenize(series[0])
            at_tokens = self.at_tokenizer.tokenize(series[1])

            smiles_tokens.insert(0, BOS_TOKEN)
            smiles_tokens.append(EOS_TOKEN)
            at_tokens.insert(0, BOS_TOKEN)
            at_tokens.append(EOS_TOKEN)

            smiles_tokens_tensor = torch.as_tensor(self.smiles_vocab.lookup_indices(smiles_tokens))
            at_tokens_tensor = torch.as_tensor(self.at_vocab.lookup_indices(at_tokens))

            seq_tensors.append((smiles_tokens_tensor, at_tokens_tensor))

        print('Padding SMILES..')
        self.smiles_padded_seqs = nn.utils.rnn.pad_sequence(
            [tuple[0] for tuple in tqdm(seq_tensors)],
            batch_first=False,
            padding_value=self.smiles_vocab.lookup_indices([PAD_TOKEN])[0]
        )

        print('Padding AT..')
        self.at_padded_seqs = nn.utils.rnn.pad_sequence(
            [tuple[1] for tuple in tqdm(seq_tensors)],
            batch_first=False,
            padding_value=self.smiles_vocab.lookup_indices([PAD_TOKEN])[0]
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        output = (self.smiles_padded_seqs[:, idx], self.at_padded_seqs[:, idx])
        return output


# In[15]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


# In[16]:


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long())


# In[17]:


class ATTransformer(pl.LightningModule):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 train_dataset: CustomDataset = None,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        if train_dataset is None:
            self.train_dataset = CustomDataset('./data/data.csv')
        else:
            self.train_dataset = train_dataset

        self.save_hyperparameters()

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.generator = nn.Sequential(
            nn.Linear(emb_size, tgt_vocab_size),
            nn.LogSoftmax(dim=2)
        )

        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size)

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")

    def forward(self, src: Tensor, tgt: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb)
        return self.generator(outs)

    def _calculate_loss(self, batch, mode="train"):
        X, y = batch
        X_hat = self.forward(X, y)

        # Calculate the index of the most probable class
        # _, X_hat = torch.max(X_hat, dim=2)

        X_hat = torch.transpose(X_hat, 1, 2)

        # print('X_hat:', X_hat.size())
        # print('X', X.size())
        # print('y', y.size())
        # print('X_hat')
        # print(X_hat)
        # print('y')
        # print(y)

        loss_f = nn.NLLLoss()
        loss = loss_f(X_hat, y)
        # Logging to WANDB
        self.log(f"{mode}_loss", loss)
        # self.log(f"{mode}_chrf_score", metrics.chrf_score(X_hat, y))
        self.log(f"{mode}_f1_score", metrics.f1_score(X_hat, y, mdmc_average='samplewise'))
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=16, shuffle=True, num_workers=2)

    # def val_dataloader(self):
        # return data.DataLoader(self.train_dataset, batch_size=64, shuffle=True, num_workers=2)


# In[18]:


# train.py
def main(hparams):
    wandb.finish()
    wandb_logger = WandbLogger(project="bachelor")

    print('Loading data..')
    dataset = CustomDataset('./data/data.csv')

    print('SMILES Vocab Size: ', dataset.smiles_vocab_len)
    print('AT Vocab Size: ', dataset.at_vocab_len)

    model = ATTransformer(
        num_encoder_layers=3,
        num_decoder_layers=3,
        emb_size=512,
        nhead=8,
        src_vocab_size=dataset.smiles_vocab_len,
        tgt_vocab_size=dataset.at_vocab_len,
        train_dataset=dataset,
        dim_feedforward=512,
        dropout=0.1)

    # train the model
    trainer = pl.Trainer(
        devices=4,
        accelerator="gpu",
        strategy='ddp',
        # precision=16,
        max_epochs=5,
        min_epochs=1,
        # overfit_batches=1,
        logger=wandb_logger
     )

    trainer.fit(model=model)


if __name__ == "__main__":
    if not in_ipython():
        root_dir = os.path.dirname(os.path.realpath(__file__))
        parser = ArgumentParser(add_help=False)
        hyperparams = parser.parse_args()

        # TRAIN
        main(hyperparams)
    else:
        main(None)

