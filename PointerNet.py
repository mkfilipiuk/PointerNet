from scipy.special import comb, factorial

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, hidden_dim, temperature=1, clipping=0):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.clipping = clipping

        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        nn.init.uniform_(self.V, -1, 1)
        
    
    def forward(self, encoder_outputs, decoder_output, mask):
        vector_of_u = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_output)) @ self.V
        masked_u = torch.where(mask, vector_of_u, -1e4)
        if self.clipping:
            return F.softmax(self.clipping*torch.tanh(masked_u), dim=1)
        return F.softmax(masked_u/self.temperature, dim=1)

class PointerNet(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=512, sampling=False, glimpse_number=0, temperature=1, clipping=0):
        super(PointerNet, self).__init__()
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
        
        self.sampling = sampling
        
        self.glimpses_and_attention = [Attention(self.hidden_dim) for i in range(glimpse_number)]
        self.attention = Attention(self.hidden_dim, temperature=1, clipping=0)
        self.glimpses_and_attention.append(self.attention)

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
            glimpse = decoder_output_n_1
            for attention in self.glimpses_and_attention[:-1]:
                glimpse = torch.sum(self.attention(seq_unpacked, glimpse, mask_n).reshape(batch_size, max_sequence_length, 1)*seq_unpacked, dim=1).reshape(batch_size, 1, self.hidden_dim)
            input_probabilities_n_1 = self.attention(seq_unpacked, glimpse, mask_n)
            outputs.append(input_probabilities_n_1)
        
            if self.sampling:
                categorical = torch.distributions.categorical.Categorical(input_probabilities_n_1)
                indices = categorical.sample()
            else:
                indices = torch.argmax(input_probabilities_n_1, 1)
            all_indices.append(indices)
            
            mask_n_1 = mask_n.clone()
            mask_n_1[torch.arange(start=0, end=batch_size, dtype=torch.long), indices] = False
            mask_n = mask_n_1
            decoder_input_n = embedded_inputs[torch.arange(start=0, end=batch_size, dtype=torch.long), indices].view(batch_size, 1, -1)
            decoder_hidden_n = decoder_hidden_n_1

        return torch.stack(outputs), torch.stack(all_indices).T
    
    def infer(self, inputs, sequence_lengths, max_sequence_length, mode="greedy", original_width=1):
        width = original_width
        is_greater = False
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
        
        if mode == "sampling":
            decoder_input_n = self.decoder_input0.repeat(batch_size*original_width, 1, 1).view(batch_size*original_width, 1, -1)
            decoder_hidden_n = (
                decoder_hidden0[0].reshape(1, batch_size, 1, -1).repeat(1, 1, original_width, 1).reshape(1, batch_size*original_width, -1), 
                decoder_hidden0[1].reshape(1, batch_size, 1, -1).repeat(1, 1, original_width, 1).reshape(1, batch_size*original_width, -1)
            )
            mask_n = (inputs > 0)[:,:,0].reshape(batch_size, 1, -1).repeat(1, original_width, 1).reshape(batch_size*original_width, -1)
            outputs = []
            all_indices = []
            for i in range(max_sequence_length):
                decoder_output_n_1, decoder_hidden_n_1 = self.decoder(
                    decoder_input_n,
                    decoder_hidden_n
                )

                glimpse = decoder_output_n_1
                seq_unpacked_expanded = seq_unpacked.reshape(batch_size, 1, max_sequence_length, -1).repeat(1, original_width, 1, 1).reshape(batch_size*original_width, max_sequence_length, -1)
                for attention in self.glimpses_and_attention[:-1]:
                    glimpse = torch.sum(self.attention(seq_unpacked_expanded, glimpse, mask_n)*seq_unpacked, dim=-1)
                input_probabilities_n_1 = self.attention(
                    seq_unpacked_expanded, 
                    glimpse, 
                    mask_n
                )
                outputs.append(input_probabilities_n_1.reshape(batch_size, original_width, -1))
                
                categorical = torch.distributions.categorical.Categorical(input_probabilities_n_1)
                indices_n_1 = categorical.sample()
                all_indices.append(indices_n_1.reshape(batch_size, original_width, -1))

                mask_n_1 = mask_n.clone()
                mask_n_1[torch.arange(start=0, end=batch_size*original_width, dtype=torch.long), indices_n_1] = False
                mask_n = mask_n_1
                decoder_input_n = embedded_inputs[torch.arange(start=0, end=batch_size, dtype=torch.long).reshape(batch_size,1).repeat(1, original_width).reshape(-1), indices_n_1].view(batch_size*original_width, 1, -1)
                decoder_hidden_n = decoder_hidden_n_1

            return torch.stack(outputs), torch.cat(all_indices, dim=-1)
            
        if mode == "greedy":
            width = 1
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
                glimpse = decoder_output_n_1
                seq_unpacked_expanded = seq_unpacked.reshape(batch_size, 1, max_sequence_length, -1).repeat(1, width, 1, 1).reshape(batch_size*width, max_sequence_length, -1)
                for attention in self.glimpses_and_attention[:-1]:
                    glimpse = torch.sum(self.attention(seq_unpacked_expanded, glimpse, mask_n)*seq_unpacked, dim=-1)
                input_probabilities_n_1 = self.attention(
                    seq_unpacked_expanded, 
                    glimpse, 
                    mask_n
                )
                outputs.append(input_probabilities_n_1.reshape(batch_size, -1, 1, max_sequence_length))
                outputs = torch.cat(outputs, dim=2)

                if (not is_greater) and original_width > comb(max_sequence_length, i+1, exact=True)*factorial(i+1, exact=True):
                    old_width = width
                    width = comb(max_sequence_length, i+1, exact=True)*factorial(i+1, exact=True)
                else:
                    old_width = width
                    width = original_width
                    is_greater = True

                if i == 0:
                    probabilities_n, indices_n = torch.topk(input_probabilities_n_1, width, dim=1)
                    probabilities_n_1, indices_n_1 = probabilities_n, indices_n
                    rows = torch.zeros((batch_size, width))
                    all_indices = [indices_n_1.reshape(batch_size, width, -1)]
                else:   
                    probabilities_n_1, indices_n_1 = torch.topk((input_probabilities_n_1.reshape(batch_size,old_width,-1)*probabilities_n.reshape(batch_size, old_width, 1)).reshape(batch_size,-1), width, dim=1)
                    probabilities_n = probabilities_n_1 / (torch.max(probabilities_n_1, dim=1).values.reshape(-1,1))
                    rows = torch.div(indices_n_1, max_sequence_length, rounding_mode="floor")
                    columns = torch.remainder(indices_n_1, max_sequence_length)
                    indices_n_1 = columns
                    all_indices = [all_indices[torch.arange(start=0, end=batch_size, dtype=torch.long).reshape(-1,1).repeat(1,width).reshape(-1), rows.reshape(-1).long()].reshape(batch_size, width, -1), indices_n_1.reshape(batch_size, width, -1)]
                outputs = [outputs[torch.arange(start=0, end=batch_size, dtype=torch.long).reshape(-1,1).repeat(1,width).reshape(-1), rows.reshape(-1).long()].reshape(batch_size, width, -1, max_sequence_length)]
                all_indices = torch.cat(all_indices, dim=2)

                if i == 0:
                    s = mask_n.shape
                    mask_n_1 = mask_n.reshape(s[0], 1, s[1]).repeat(1, width, 1).reshape(s[0]*width,s[1]).clone()
                    mask_n_1[torch.arange(start=0, end=batch_size*width, dtype=torch.long), indices_n_1.reshape(-1)] = False
                    decoder_input_n = embedded_inputs[torch.arange(start=0, end=batch_size, dtype=torch.long).reshape(-1,1).repeat(1,width).reshape(-1), indices_n_1.reshape(-1)].view(batch_size*width, 1, -1)

                    decoder_hidden_n = (decoder_hidden_n_1[0].repeat(1, width, 1), decoder_hidden_n_1[1].repeat(1, width, 1))
                else:
                    mask_n_1 = mask_n.reshape(batch_size, old_width, -1)[torch.arange(start=0, end=batch_size, dtype=torch.long).reshape(-1,1).repeat(1,width).reshape(-1), rows.reshape(-1)].clone()
                    mask_n_1[torch.arange(start=0, end=batch_size*width, dtype=torch.long), columns.reshape(-1)] = False
                    decoder_input_n = embedded_inputs.reshape(batch_size, 1, max_sequence_length, -1).repeat(1, width,1,1)[torch.arange(start=0, end=batch_size, dtype=torch.long).reshape(-1,1).repeat(1,width).reshape(-1), rows.reshape(-1), columns.reshape(-1)].view(batch_size*width, 1, -1)
                    decoder_hidden_n = (decoder_hidden_n_1[0].reshape(batch_size, old_width, -1)[torch.arange(start=0, end=batch_size, dtype=torch.long).reshape(-1,1).repeat(1,width).reshape(-1), rows.reshape(-1)].view(1, batch_size*width, -1), decoder_hidden_n_1[1].reshape(batch_size, old_width, -1)[torch.arange(start=0, end=batch_size, dtype=torch.long).reshape(-1,1).repeat(1,width).reshape(-1), rows.reshape(-1)].view(1, batch_size*width, -1))
                mask_n = mask_n_1.reshape(batch_size*width, -1)

            return torch.stack(outputs), all_indices