# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn # type: ignore
from transformers import RobertaModel # type: ignore

class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.code_encoder = encoder
        self.nl_encoder = RobertaModel.from_pretrained(encoder.config._name_or_path)  # Clone for NL
        
    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            return self.code_encoder(code_inputs, attention_mask=code_inputs.ne(1))[1]
        else:
            return self.nl_encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]