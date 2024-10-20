from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisonConfig:
    def __init__(
            self,
            hidden_size: int = 768,
            intermedia_size=3072,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 12,
            num_channels: int = 3,
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens:int=None,
            **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermedia_size = intermedia_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self,config:SiglipVisonConfig):
        super().__init__()
        self.config=config
        self.embed_dim=config.hidden_size
        self.image_size=config.image_size
        self.patch_size=config.patch_size
        
        self.patch_embeddings=nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicates no padding is added
        )
        self.num_patches=(self.image_size//self.patch_size)**2
        self.num_positons=self.num_patches
        self.position_embeddings=nn.Embedding(self.num_positons,self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positons).expand(1,-1),
            persistent=False,

        )

class SiglipVisonTransformer(nn.Module):
    def __iniit__(self,config:SiglipVisonConfig):
        super().__init__()
        self.config=config
        embed_dim=config.hidden_size

        self.embeddings=SiglipVisionEmbeddings(config)
        self.encoder=SiglipEncoder(config) 
        self.post_layernorm=nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)

    def forward(self,pixel_values:torch.Tensor)->torch.Tensor:
        # pixel_values: [Batch_Size,Channels,Height,Width] -> [Batch_size,Num_Patches,Embedding_Dim]
        hidden_states=self.embeddings(pixel_values)

        last_hidden_state=self.encoder(inputs_embeds=hidden_states)
        last_hidden_state=self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisonModel(nn.Module):
    
    def __init__(self,config:SiglipVisonConfig):
        super().__init__()
        self.config = config
        self.vision_model=SiglipVisonTransformer(config)

    def forward(self,pixel_values)-> Tuple:
        # [Batch_Size,Channels,Height,Width] -> [Batch_Size,Num_Patches,Embd_Dim]
        return self.vision_model(pixel_values=pixel_values)

