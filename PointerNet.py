import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        nn.init.uniform_(self.V, -1, 1)
        
    
    def forward(self, encoder_outputs, decoder_output, mask):
        vector_of_u = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_output)) @ self.V
        masked_u = torch.where(mask, vector_of_u, -1e4)
        return F.softmax(masked_u, dim=1)

class PointerNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, lstm_layers=1):
        super(PointerNet, self).__init__()
        # encoder
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
        
        # decoder
        self.decoder_input0_weight = torch.FloatTensor(embedding_dim)
        self.decoder_input0 = torch.nn.parameter.Parameter(self.decoder_input0_weight)
        nn.init.uniform_(self.decoder_input0, -0.08, 0.08)
        
        self.decoder = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            lstm_layers,
            batch_first=True
        )
        
        self.attention = Attention(self.hidden_dim)

    def forward(self, inputs, sequence_lengths):
        batch_size = sequence_lengths.shape[0]
        max_sequence_length = torch.max(sequence_lengths).item()
        # input embedding 
        embedded_inputs = self.embedding(inputs)
        
        # encoder
        encoder_outputs, encoder_hidden = self.encoder(
            torch.nn.utils.rnn.pack_padded_sequence(embedded_inputs, sequence_lengths, batch_first=True, enforce_sorted=False), 
            (self.encoder_h0.repeat(batch_size, 1).view(1, batch_size, -1), self.encoder_c0.repeat(batch_size, 1).view(1, batch_size, -1))
        )
        decoder_hidden0 = (encoder_hidden[0][-1].view(1, batch_size, -1), encoder_hidden[1][-1].view(1, batch_size, -1))
        
        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True, total_length=max_sequence_length)
        
        # decoder
        
        decoder_input_n = self.decoder_input0.repeat(batch_size, 1, 1).view(batch_size, 1, -1)
        decoder_hidden_n = decoder_hidden0
        mask_n = (inputs > 0)[:,:,0]
        outputs = []
        all_indices = []
        for i in range(max_sequence_length):
            decoder_output_n_1, decoder_hidden_n_1 = self.decoder(
                decoder_input_n,
                decoder_hidden_n
            )
            
            input_probabilities_n_1 = self.attention(seq_unpacked, decoder_output_n_1, mask_n)
            outputs.append(input_probabilities_n_1)
        
            indices = torch.argmax(input_probabilities_n_1, 1)
            all_indices.append(indices)
            
            mask_n_1 = mask_n.clone()
            mask_n_1[torch.arange(start=0, end=batch_size, dtype=torch.long), indices] = False
            mask_n = mask_n_1
            decoder_input_n = embedded_inputs[torch.arange(start=0, end=batch_size, dtype=torch.long), indices].view(batch_size, 1, -1)
            decoder_hidden_n = decoder_hidden_n_1

        return torch.stack(outputs), torch.stack(all_indices).T