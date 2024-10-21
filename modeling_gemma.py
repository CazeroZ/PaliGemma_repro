import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig,SiglipVisionModel



class GemmaConfig():
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size=vocab_size
        self.max_position_embeddings=max_position_embeddings
        self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.num_key_value_heads=num_key_value_heads
        self.head_dim=head_dim
        self.rms_norm_eps=rms_norm_eps
        self.rope_theta=rope_theta
        self.attention_bias=attention_bias
        self.attetion_dropout=attention_dropout
        self.pad_token_id=pad_token_id

class PaliGemmaConfig:
    def __init__(
        self,
        vision_config=None,
        text_config=None,   
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index=ignore_index
        self.image_token_index=image_token_index
        self.projection_dim=projection_dim
        self.hidden_size=hidden_size
        self.vision_conifg=vision_config
        self.is_encoder_decoder=False
        self.pad_token_id=pad_token_id
        

        self.vision_conifg=SiglipVisionConfig(**vision_config)
        self.text_conig=text_config

        self.text_conig=GemmaConfig(**text_config,pad_token_id=pad_token_id)
        self.vocab_size=self.text_conig.vocab_size

        self.text_conig.num_image_tokens=(self.vision_conifg.image_size//self.vision_conifg.patch_size)**2
        self.vision_conifg.projection_dim=projection_dim

class GemmaRMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-6):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.zeros(dim))

    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    
    
    def forward(self,x):
        output=self._norm(x.float())
        output=output*(1.0+self.weight.float())
        return output.type_as(x)

class GemmaMLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.hidden_size=config.hidden_size
        self.intermediate_size=config.intermediate_size
        self.gate_proj=nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.up_proj=nn.Linear(self.intermediate_size,self.hidden_size,bias=False)
        self.down_proj=nn.Linear(self.intermediate_size,self.hidden_size,bias=False)
    
    def forward(self,x):
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]

        return self.down_proj(nn.functional.gelu(self.gate_proj(x),approximate="tanh")*self.up_proj(x))

# TODO : Implement the GemmaAttenion class

class GemmaDecoderLayer(nn.Module):
    def __init__(self,config:GemmaConfig,layer_idx:int):
        super().__init__()
        self.hidden_size=config.hidden_size
        self.self_attn=GemmaAttention(config,layer_idx=layer_idx)

        self.mlp=GemmaMLP(config)
        self.input_layernorm=GemmaRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm=GemmaRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states:torch.Tensor,
        attention_mask:Optional[torch.Tensor]=None,
        position_ids:Optional[torch.LongTensor]=None,
        kv_cache:Optional[KVCache]=None,
    )-> Tuple[torch.FloatTensor,Optional[Tuple[torch.FloatTensor,torch.FloatTensor]]]:
        residual=hidden_states
        #[Batch_Size,Seq_Len,Hidden_Size]
        hidden_states=self.input_layernorm(hidden_states)   
        hidden_states,_,=self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,

        )
        #[Batch_szie,Seq_Len,Hidden_Size]
        hidden_states=hidden_states+residual

        # [Batch_Size, Seq_Len, Hidden_Size]
        residual=hidden_states

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states=self.post_attention_layernorm(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states=self.mlp(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states=residual+hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self,config:GemmaConfig):
        super().__init__()
        self.padding_idx=config.pad_token_id
        self.vocab_size=config.vocab_size

        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size,self.padding_idx)
        self.layers=nn.ModuleList(
            [GemmaDecoderLayer(config,layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm=GemmaRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        attention_mask:Optional[torch.Tensor]=None,
        position_ids:Optional[torch.LongTensor]=None,
        inputs_embeds:Optional[torch.FloatTensor]=None,
        kv_cache:Optional[KVCache]=None,
    )->torch.FloatTensor:
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states=inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        normalizer=torch.tensor(self.config.hidden_size**0.5,dtype=hidden_states.dtype)
        hidden_states=hidden_states*normalizer

        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states=decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states=self.norm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        return hidden_states
class GemmaForCausalLM(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.model=GemmaModel(config)
        self.vocab_size=config.vocab_size
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    def tie_weights(self):
        self.lm_head.weight=self.modelf.embed_tokens.weight
    def forward(
        self,
        attention_mask:Optional[torch.Tensor]=None,
        position_ids:Optional[torch.LongTensor]=None,
        inputs_embeds:Optional[torch.FloatTensor]=None,
        kv_cache:Optional[KVCache]=None,
    )-> Tuple:
        # input_embeds: [Batch_Size,Seq_len,Hidden_Size]
        # outputs: [Batch_Size,Seq_len,Vocab_Size]
        outputs=self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        
        hidden_states=outputs
        logits=self.lm_head(hidden_states)
        logits=logits.float()

        return_data={
            "logits": logits
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"]=kv_cache
        return return_data


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_conifg.hidden_size, config.vision_conifg.projection_dim, bias=True)

    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Hidden_Size] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states



 

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self,config:PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower=SiglipVisionModel(config.vision_config)
        self.multi_modal_projector=PaliGemmaMultiModalProjector(config)
        self.vocab_size=config.vocab_size

        language_model=GemmaForCausalLM(config.text_config)
        self.language_model=language_model

        self.pad_token_id=self.config.pad_token_id if self.config.pad_tokne_id is not None else -1
        
    
    def tie_weights(self):
        return self.language_model.tie_weights()
    def _merge_input_ids_with_image_features(
        self,image_features:torch.Tensor,inputs_embeds:torch.Tensor,input_ids:torch.Tensor,attention_mask:torch.Tensor,kv_cache:Optional[KVCache]=None,
    ):
        _,_,embed_dim=image_features.shape
        batch_size,sequence_len=input_ids.shape
        dtype,device=inputs_embeds.dtype,inputs_embeds.device()
        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features=image_features/(self.confg.hidden_size**0.5)

        # Combine teh embeddings of the image tokens , the text tokens and the mask  out all the padding tokens
        final_embedding=torch.zeros(batch_size,sequence_len,embed_dim,dtype=dtype,device=inputs_embeds.device)
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask=(input_ids!=self.config.image_token_index)& (input_ids!=self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask=(input_ids==self.config.image_token_index)
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask=input_ids==self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded=text_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
        image_mask_expanded=image_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
        pad_mas_expanded=pad_mask.unsqueeze(-1).expand(-1,-1,embed_dim)

        final_embedding=torch.where(text_mask_expanded,inputs_embeds,final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of  scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding=torch.masked_scatter(image_mask_expanded,scaled_image_features)
        # Zero out padding tokens
        final_embedding=torch.where(pad_mask,torch.zeros_like(final_embedding),final_embedding)
        
        #TODO : Implement the rest of the function

        ### CREATE The ATTENTION MASK ###
        dtype,device=inputs_embeds.dtype, inputs_embeds.device
        min_dtype=torch.finfo(dtype).min
        q_len=inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items()==0:
            # Do not mask any token, 
            casual_mask=torch.full(
                (batch_size,q_len,q_len),fill_value=0,dtype=dtype,device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len==1
            kv_len=kv_cache.num_items()+q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            casual_mask=torch.full(
                (batch_size,q_len,kv_len),fill_value=0,dtype=dtype,device=device
            ) 

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1) 
        
        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids=attention_mask.cumsum(-1)[:,-1]
            if position_ids.dim()==1:
                position_ids=position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position
            position_ids=(attention_mask.cumsum(-1)).masked_fill((attention_mask==0),1).to(device)

        return final_embedding, causal_mask, position_ids
    def forward(
        self,
        input_ids: torch.LongTensor=None,
        pixel_values:torch.FloatTensor=None,
        attention_mask:Optional[torch.Tensor]=None,
        kv_cache:Optional[KVCache]=None,
    ) -> Tuple:
        assert torch.all(attention_mask==1),"The input cannot be padded"

        #1. Extract the input embeddings
        # shape: (Batch_szie,Seq_len, Hidden_Size)
        inputs_embeds=self.language_model.get_input_embeddings()(input_ids)

        #2. Merge the text and images
        #[Batch_Size, Channels,Height,Width]-> [Batch_size,Num_Patches,Embed_Dim]
        selected_image_feature=self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # [Batch_Size,Num_Patches,Embed_Dim]->[Batch_Size,Num_Patches,Hidden_Size]
        image_features=self.multi_modal_projector(selected_image_feature)

        # Merge the embeddings of the text tokens and image tokens
        inputs_embeds, attention_mask,position_ids=self._merge_input_ids_with_image_features(image_features,inputs_embeds,input_ids,attention_mask,kv_cache)

        outputs=self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,

        )
        
        return  outputs