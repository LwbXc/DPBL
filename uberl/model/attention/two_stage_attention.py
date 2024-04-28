import torch.nn as nn
import torch
import pdb

from .multi_head import MultiHeadedAttention


class ExtractorAndDenoising(nn.Module):

    def __init__(self, hidden, attn_heads, preference_num):

        super().__init__()
        self.attention_extractor = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.attention_denoising = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.preferences_mu = nn.Parameter(torch.randn(preference_num, hidden))
        self.preferences_sigma = nn.Parameter(torch.randn(preference_num, hidden))
        self.preference_num = preference_num

    def forward(self, x, mask):
        '''The proposed two stage attention'''
        mask = mask.float()
        mask1 = mask.unsqueeze(1).repeat(1, self.preference_num, 1).unsqueeze(1)
        mask2 = mask1.permute(0, 1, 3, 2)


        preference_distributions = torch.distributions.Normal(loc=self.preferences_mu, scale=torch.exp(self.preferences_sigma))
        preferences = preference_distributions.rsample()

        hidden_states, attn_1 = self.attention_extractor.forward(preferences.unsqueeze(0).repeat(x.size(0), 1, 1), x, x, mask=mask1)
        noise, other_preferences = hidden_states[:, 0:1, :], hidden_states[:, 1:, :]
        zeros = torch.zeros_like(noise)
        hidden_values = torch.cat([zeros, other_preferences], dim=1)

        output, attn_2 = self.attention_denoising.forward(x, hidden_states, hidden_values, mask2) 
        return output, attn_1, attn_2, hidden_states
