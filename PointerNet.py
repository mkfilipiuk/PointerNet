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
    def __init__(self, embedding_dim=128, hidden_dim=512):
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
        
        self.attention = Attention(self.hidden_dim)

    def forward(self, inputs, sequence_lengths, max_sequence_length, mode="greedy", original_beam_search_width=-1):
        beam_search_width = original_beam_search_width
        is_greater = False
        batch_size = sequence_lengths.shape[0]
        # input embedding 
        embedded_inputs = self.embedding(inputs)
        
        # encoder
        encoder_outputs, encoder_hidden = self.encoder(
            torch.nn.utils.rnn.pack_padded_sequence(embedded_inputs, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False), 
            (self.encoder_h0.repeat(batch_size, 1).view(1, batch_size, -1), self.encoder_c0.repeat(batch_size, 1).view(1, batch_size, -1))
        )
        decoder_hidden0 = (encoder_hidden[0][-1].view(1, batch_size, -1), encoder_hidden[1][-1].view(1, batch_size, -1))
        
        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True, total_length=max_sequence_length)
        
        # decoder
        if mode == "beam_search":
            if (not is_greater) and original_beam_search_width > max_sequence_length:
                beam_search_width = max_sequence_length
            else:
                beam_search_width = original_beam_search_width
                is_greater = True
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
            input_probabilities_n_1 = self.attention(seq_unpacked.repeat(beam_search_width if i > 0 and original_beam_search_width > 0 else 1,1,1), decoder_output_n_1, mask_n)
            outputs.append(input_probabilities_n_1)
            
            if mode == "greedy":
                indices_n_1 = torch.argmax(input_probabilities_n_1, 1)
            if mode == "sampling":
                categorical = torch.distributions.categorical.Categorical(input_probabilities_n_1)
                indices_n_1 = categorical.sample()
            if mode == "beam_search":
                if i == 0:
                    probabilities_n, indices_n = torch.topk(input_probabilities_n_1, beam_search_width, dim=1)
                    probabilities_n_1, indices_n_1 = probabilities_n, indices_n
                else:
                    if mode == "beam_search":
                        if (not is_greater) and original_beam_search_width > max_sequence_length**(i+1):
                            old_beam_search_width = beam_search_width
                            beam_search_width = max_sequence_length**(i+1)
                        else:
                            old_beam_search_width = beam_search_width
                            beam_search_width = original_beam_search_width
                            is_greater = True
                    probabilities_n_1, indices_n_1 = torch.topk((input_probabilities_n_1.reshape(batch_size,old_beam_search_width,-1)*probabilities_n.reshape(batch_size, old_beam_search_width, 1)).reshape(batch_size,-1), beam_search_width, dim=1)
                    probabilities_n = probabilities_n_1 / (torch.max(probabilities_n_1, dim=1).values.reshape(-1,1))
                    rows = torch.div(indices_n_1, max_sequence_length, rounding_mode="floor")
                    columns = torch.remainder(indices_n_1, max_sequence_length)
                    indices_n_1 = columns
            all_indices.append(indices_n_1)

            if mode == "beam_search" and i == 0:
                s = mask_n.shape
                mask_n_1 = mask_n.reshape(s[0], 1, s[1]).repeat(1, beam_search_width, 1).reshape(s[0]*beam_search_width,s[1]).clone()
                mask_n_1[torch.arange(start=0, end=batch_size*beam_search_width, dtype=torch.long), indices_n_1.reshape(-1)] = False
                decoder_input_n = embedded_inputs.repeat(beam_search_width, 1, 1)[torch.arange(start=0, end=batch_size*beam_search_width, dtype=torch.long), indices_n_1.reshape(-1)].view(batch_size*beam_search_width, 1, -1)

                decoder_hidden_n = (decoder_hidden_n_1[0].repeat(1, beam_search_width, 1), decoder_hidden_n_1[1].repeat(1, beam_search_width, 1))
            elif mode == "beam_search" and i > 0:
                mask_n_1 = mask_n.reshape(batch_size, old_beam_search_width, -1)[torch.arange(start=0, end=batch_size, dtype=torch.long).reshape(-1,1).repeat(1,beam_search_width).reshape(-1), rows.reshape(-1)].clone()
                mask_n_1[torch.arange(start=0, end=batch_size*beam_search_width, dtype=torch.long), columns.reshape(-1)] = False
                decoder_input_n = embedded_inputs.reshape(batch_size, 1, max_sequence_length, -1).repeat(1, beam_search_width,1,1)[torch.arange(start=0, end=batch_size, dtype=torch.long).reshape(-1,1).repeat(1,beam_search_width).reshape(-1), rows.reshape(-1), columns.reshape(-1)].view(batch_size*beam_search_width, 1, -1)
                decoder_hidden_n = (decoder_hidden_n_1[0].reshape(batch_size, old_beam_search_width, -1)[torch.arange(start=0, end=batch_size, dtype=torch.long).reshape(-1,1).repeat(1,beam_search_width).reshape(-1), rows.reshape(-1)].view(1, batch_size*beam_search_width, -1), decoder_hidden_n_1[1].reshape(batch_size, old_beam_search_width, -1)[torch.arange(start=0, end=batch_size, dtype=torch.long).reshape(-1,1).repeat(1,beam_search_width).reshape(-1), rows.reshape(-1)].view(1, batch_size*beam_search_width, -1))
            else:
                mask_n_1 = mask_n.clone()
                mask_n_1[torch.arange(start=0, end=batch_size, dtype=torch.long), indices_n_1] = False
                decoder_input_n = embedded_inputs[torch.arange(start=0, end=batch_size, dtype=torch.long), indices_n_1].view(batch_size, 1, -1)
                decoder_hidden_n = decoder_hidden_n_1
            mask_n = mask_n_1.reshape(batch_size*beam_search_width if beam_search_width > 0 else batch_size, -1)
            


        import pdb; pdb.set_trace()
        return torch.stack(outputs), torch.stack(all_indices).T