from scipy.special import comb, factorial

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from models.PointerNet.PointerNet import Attention

class CriticNet(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=128, p=3, d=64):
        super(CriticNet, self).__init__()
        # encoder
        lstm_layers = 1
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(2, embedding_dim)
        self.encoder_h0_weight = torch.FloatTensor(hidden_dim)   
        self.encoder_c0_weight = torch.FloatTensor(hidden_dim)
        self.encoder_h0 = torch.nn.parameter.Parameter(self.encoder_h0_weight)
        self.encoder_c0 = torch.nn.parameter.Parameter(self.encoder_c0_weight)
        nn.init.uniform_(self.encoder_h0, -0.08, 0.08)
        nn.init.uniform_(self.encoder_c0, -0.08, 0.08)
        self.encoder = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            lstm_layers,
            batch_first=True
        )
        
        # process block
        self.decoder_input0_weight = torch.FloatTensor(hidden_dim)
        self.decoder_input0 = torch.nn.parameter.Parameter(self.decoder_input0_weight)
        nn.init.uniform_(self.decoder_input0, -0.08, 0.08)
        
        self.process_block = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            lstm_layers,
            batch_first=True
        )
        
        self.attention = Attention(self.hidden_dim, temperature=1, clipping=0)
        
        # decoder
        self.d = d
        self.p = p
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, 1),
        )

    def forward(self, inputs, sequence_lengths, max_sequence_length):
        batch_size = sequence_lengths.shape[0]
        # input embedding 
        embedded_inputs = self.embedding(inputs)
        
        # encoder
        encoder_outputs, encoder_hidden = self.encoder(
            torch.nn.utils.rnn.pack_padded_sequence(
                embedded_inputs, 
                sequence_lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            ), (
                self.encoder_h0.repeat(batch_size, 1).view(1, batch_size, -1), 
                self.encoder_c0.repeat(batch_size, 1).view(1, batch_size, -1)
            )
        )
        decoder_hidden0 = (
            encoder_hidden[0][-1].view(1, batch_size, -1), 
            encoder_hidden[1][-1].view(1, batch_size, -1)
        )
        
        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(
            encoder_outputs, 
            batch_first=True, 
            total_length=max_sequence_length
        )
        
        # process block
        
        decoder_input_n = self.decoder_input0.repeat(batch_size, 1, 1).view(batch_size, 1, -1)
        decoder_hidden_n = decoder_hidden0
        mask = (inputs > 0)[:,:,0]
        for i in range(self.p):
            decoder_output_n_1, decoder_hidden_n_1 = self.process_block(
                decoder_input_n,
                decoder_hidden_n
            )
            glimpse = torch.sum(self.attention(seq_unpacked, decoder_output_n_1, mask).reshape(batch_size, max_sequence_length, 1)*seq_unpacked, dim=1).reshape(batch_size, 1, self.hidden_dim)
            decoder_input_n = glimpse
            decoder_hidden_n = decoder_hidden_n_1
        decoder_output_n_1, decoder_hidden_n_1 = self.process_block(
            decoder_input_n,
            decoder_hidden_n
        )
            
        # decoder
        return self.decoder(decoder_output_n_1).reshape(batch_size)
