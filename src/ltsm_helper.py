import numpy as np
import json

#word2vec
import gensim
import torch
import torch.nn as nn
from dataclasses import dataclass


WORD_FREQ = 20
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128
ATTENTION_SIZE = 32
SEQ_LEN = 128
TEST_SIZE = 0.1
BATCH_SIZE = 1024
DEVICE = 'cuda'
EPOCHS = 20


@dataclass
class ConfigRNN:
    vocab_size: int
    batch_size: int
    device: str
    n_layers: int
    embedding_dim: int
    hidden_size: int
    seq_len: int
    bidirectional: bool
    attention_size: int
    dropout: float
    output: int
    
    
class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super(BahdanauAttention, self).__init__()
        # Linear layers to transform encoder outputs and decoder hidden state
        self.W1 = nn.Linear(encoder_dim, decoder_dim)
        self.W2 = nn.Linear(decoder_dim, decoder_dim)
        # v is a learnable parameter vector to compute attention scores
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self,encoder_outputs, hidden):
        # hidden: (batch_size, decoder_dim)
        # encoder_outputs: (batch_size, seq_len, encoder_dim)
        seq_len = encoder_outputs.size(1)

        # Repeat hidden state seq_len times for addition
        hidden_with_time_axis = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # score shape: (batch_size, seq_len, decoder_dim)
        score = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden_with_time_axis))
        # attention_weights shape: (batch_size, seq_len, 1)
        attention_weights = torch.softmax(self.v(score), dim=1)

        # context_vector: weighted sum of encoder outputs
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)
        # attention_weights: (batch_size, seq_len)
        return attention_weights.squeeze(-1), context_vector


def padding(reviews, seq_len):
    features = np.zeros((len(reviews), seq_len))
    for i, review in enumerate(reviews):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[:seq_len]
        features[i, :] = np.array(new)
        
    return features


def tokenize_text(text):
    text_int = [vocab2int[word] for word in text.split() if vocab2int.get(word)]
    padded = padding([text_int], SEQ_LEN)
    
    return padded


f = open('./notebooks/vocab2int.json', 'r')
vocab2int = json.load(f) 
VOCAB_SIZE = len(vocab2int) + 1

wv = gensim.models.Word2Vec.load("./models/word2vec_model")
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

for word, i in vocab2int.items():
    try:
        embedding_vector = wv.wv[word]
        embedding_matrix[i] = embedding_vector
    except KeyError as e:
        print(f'{e}: word: {word}')
        pass
    
embedding_layer = nn.Embedding.from_pretrained(
    torch.FloatTensor(embedding_matrix)
)

net_config = ConfigRNN(
    vocab_size=VOCAB_SIZE,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    n_layers=3,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    seq_len=SEQ_LEN,
    bidirectional=True,
    attention_size=ATTENTION_SIZE,
    dropout=0.4,
    output=1
)

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        config=net_config
    ):
        super(LSTMClassifier, self).__init__()
        self.embedding = embedding_layer
        self.hidden_size = config.hidden_size
        self.n_layers = config.n_layers
        self.bidirectional = config.bidirectional
        self.num_directions = 2 if config.bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_size,
            num_layers=config.n_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0,
        )

        # Attention layer
        self.attention = BahdanauAttention(
            encoder_dim=config.hidden_size * self.num_directions,
            decoder_dim=config.hidden_size * self.num_directions,
        )

        # Final fully connected layer for classification
        self.fc = nn.Linear(config.hidden_size * self.num_directions, config.output)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_sequences, lengths=None):
        # input_sequences: (batch_size, seq_len)
        # lengths: list or tensor of actual sequence lengths (for packing)
        embedded = self.embedding(input_sequences)
        embedded = self.dropout(embedded)

        # Pack sequences if lengths provided
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_outputs, (hidden, cell) = self.lstm(packed)
            # Unpack
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
                packed_outputs, batch_first=True
            )
        else:
            encoder_outputs, (hidden, cell) = self.lstm(embedded)

        # Concatenate the final forward and backward hidden states
        if self.bidirectional:
            # hidden: (num_layers*2, batch, hidden_size)
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            final_hidden = hidden[-1]

        # Apply Bahdanau attention
        attn_weights, context_vector = self.attention(
            encoder_outputs, final_hidden
        )

        # Classifier
        logits = self.fc(self.dropout(context_vector))
        return logits, attn_weights