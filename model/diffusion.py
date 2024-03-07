import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Linformer as LF
#
# class Linformer_Layer(nn.Module):
#     """
#     A wrapper function to accept LM tasks, inspired by https://github.com/lucidrains/sinkhorn-transformer
#     """
#     def __init__(self, num_tokens, input_size, channels,
#                        dim_k=64, dim_ff=1024, dim_d=None,
#                        dropout_ff=0.1, dropout_tokens=0.1, nhead=4, depth=1, ff_intermediate=None,
#                        dropout=0.05, activation="gelu", checkpoint_level="C0",
#                        parameter_sharing="layerwise", k_reduce_by_layer=0, full_attention=False,
#                        include_ff=True, w_o_intermediate_dim=None, emb_dim=None,
#                        return_emb=False, decoder_mode=False, causal=False, method="learnable"):
#         super(Linformer_Layer, self).__init__()
#         emb_dim = channels if emb_dim is None else emb_dim
#
#         self.input_size = input_size
#
#         self.to_token_emb = nn.Embedding(num_tokens, emb_dim)
#         self.pos_emb = LF.PositionalEmbedding(emb_dim)
#         self.linformer = LF.Linformer(input_size, channels, dim_k=dim_k,
#                                    dim_ff=dim_ff, dim_d=dim_d, dropout_ff=dropout_ff,
#                                    nhead=nhead, depth=depth, dropout=dropout, ff_intermediate=ff_intermediate,
#                                    activation=activation, checkpoint_level=checkpoint_level, parameter_sharing=parameter_sharing,
#                                    k_reduce_by_layer=k_reduce_by_layer, full_attention=full_attention, include_ff=include_ff,
#                                    w_o_intermediate_dim=w_o_intermediate_dim, decoder_mode=decoder_mode, causal=causal, method=method)
#
#         if emb_dim != channels:
#             self.linformer = LF.ProjectInOut(self.linformer, emb_dim, channels)
#
#         self.dropout_tokens = nn.Dropout(dropout_tokens)
#
#     def forward(self, tensor, **kwargs):
#         """
#         Input is (batch_size, seq_len), and all items are ints from [0, num_tokens-1]
#         """
#         # tensor = self.to_token_emb(tensor)
#         tensor = self.pos_emb(tensor).type(tensor.type()) + tensor
#         tensor = self.dropout_tokens(tensor)
#         tensor = self.linformer(tensor, **kwargs)
#         return tensor

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        # print('x shape',x.shape) [12,320]
        # print('t shape',t.shape) [12]
        out = self.lin(x)
        gamma = self.embed(t)
        # print('gramma shape',gamma.shape) [12,320]
        out = gamma.view(-1, self.num_out) * out

        return out


class ConditionalModel(nn.Module):
    def __init__(self, timesteps=1000, feature_dim=320, num_classes=100, guidance=True):
        super(ConditionalModel, self).__init__()
        n_steps = timesteps + 1
        y_dim = num_classes
        feature_dim = feature_dim

        self.guidance = guidance
        # encoder for x
        self.encoder_x = nn.AdaptiveAvgPool1d(feature_dim)
        # batch norm layer
        self.norm = nn.BatchNorm1d(feature_dim)

        # Unet
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim * 2, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)
        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, x, y, t, yhat=None):
        x = self.encoder_x(x)
        x = self.norm(x)
        if self.guidance:
            # for yh in yhat:
            y = torch.cat([y, yhat], dim=-1)
        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = x * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        y = self.lin4(y)

        return y

if __name__ == '__main__':
    num_in = 128
    num_out = 4
    x = torch.rand((12,128))
    t = torch.randint(low=0, high=1, size=(12,))

    model = ConditionalLinear(num_in, num_out, 1)
    y = model(x,t)
    print(y.shape)