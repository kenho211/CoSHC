# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn # type: ignore
import torch # type: ignore
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
        
    def to(self, device):
        self.code_encoder.to(device)
        self.nl_encoder.to(device)
        return self


# --------------------------------------------------
# CoSHC Model Architecture (Extends Base Model)
# --------------------------------------------------
class CoSHCModel(torch.nn.Module):
    def __init__(self, base_model, hash_dim=128, num_clusters=10):
        super().__init__()
        self.base_model = base_model
        self.alpha = 1.0 # Initial alpha value for scaling factor (will increase during training)
        
        # Hashing Module (Section 3.1.2)
        self.code_hash = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, hash_dim),
            torch.nn.Tanh()
        )
        self.nl_hash = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, hash_dim),
            torch.nn.Tanh()
        )
        
        # Classification Module (Section 3.2.2)
        self.classifier = torch.nn.Linear(768, num_clusters)
        
    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            embeddings = self.base_model(code_inputs=code_inputs)
            return embeddings
        else:
            embeddings = self.base_model(nl_inputs=nl_inputs)
            return embeddings
            
    def predict_category(self, nl_inputs):
        embeddings = self.base_model(nl_inputs=nl_inputs)
        return torch.softmax(self.classifier(embeddings), dim=-1)
    

    def get_binary_hash(self, inputs, is_code=True, apply_tanh=False):
        """Get binary hash using sign function (equation 5); For inference"""
        print(inputs.device)
        print(self.code_hash[0].weight.device)
        if is_code:
            h = self.code_hash[:-1](inputs)
        else:
            h = self.nl_hash[:-1](inputs)
        
        # Apply equation 6: tanh(alpha * H)
        if apply_tanh:
            return torch.tanh(self.alpha * h)
        else:
            return torch.sign(h)  # Equation 5


    def to(self, device):
        self.base_model.to(device)

        for layer in self.code_hash:
            layer.to(device)
        for layer in self.nl_hash:
            layer.to(device)
        
        self.classifier.to(device)

        print(self.code_hash[0].weight.device)
        print(self.code_hash[2].weight.device)
        print(self.code_hash[4].weight.device)

        print(self.nl_hash[0].weight.device)
        print(self.nl_hash[2].weight.device)
        print(self.nl_hash[4].weight.device)
        
        print(self.classifier.weight.device)
        return self