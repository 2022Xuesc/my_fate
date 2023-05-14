__all__ = ['CnnRnn']

from federatedml.nn.backend.multi_label.models.lstm.encoders import EncoderCNN
from federatedml.nn.backend.multi_label.models.lstm.decoders import LabelRNN

import torch.nn as nn


class CnnRnn(nn.Module):
    def __init__(self, embed_size, hidden_size, label_num, num_layers,  device,max_seq_length=5):
        super(CnnRnn, self).__init__()
        self.encoder = EncoderCNN(embed_size).to(device)
        self.decoder = LabelRNN(embed_size, hidden_size, label_num, num_layers, max_seq_length=max_seq_length).to(
            device)

    def forward(self, images, labels, lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, labels, lengths)
        return outputs
