import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
import torch.nn as nn
from simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()



def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

class PromptLearner(nn.Module):
    def __init__(self, clip_model,n_ctx_general,n_ctx_special):
        super().__init__()
        # n_cls = len(classnames)
        self.n_ctx_general = n_ctx_general #4,8,16
        self.n_ctx_special = n_ctx_special
        
        # self.dtype = clip_model.dtype
        self.dtype = torch.float32
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if not self.n_ctx_general == 0:
            ctx_vectors = torch.empty(self.n_ctx_general, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
        if not self.n_ctx_special == 0:
            ctx_vectors0 = torch.empty(self.n_ctx_special, ctx_dim, dtype=self.dtype)
            ctx_vectors1 = torch.empty(self.n_ctx_special, ctx_dim, dtype=self.dtype)
        
            nn.init.normal_(ctx_vectors0, std=0.02)
            nn.init.normal_(ctx_vectors1, std=0.02)
        self.prompt_prefix = " ".join(["X"] * (self.n_ctx_general+self.n_ctx_special))

        if not self.n_ctx_general == 0:
            self.ctx = nn.Parameter(ctx_vectors,requires_grad=True)  # to be optimized
        else:
            self.ctx = None

        if not self.n_ctx_special == 0:
            self.ctx0 = nn.Parameter(ctx_vectors0,requires_grad=True)  # to be optimized
            self.ctx1 = nn.Parameter(ctx_vectors1,requires_grad=True)  # to be optimized
        else:
            self.ctx0 = None
            self.ctx1 = None

        self.clip_model = clip_model

        self.class_token_position = 'end' # 'end'/'middle'/"front"

    def forward(self, classnames,mode):
        n_cls = len(classnames)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts_ori = [self.prompt_prefix + " " + name + "." for name in classnames]
        # prompts_ori = [name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts_ori]).to('cuda')
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
        if mode == 0:
            ctxs = self.ctx0
        if mode == 1:
            ctxs = self.ctx1
        ctx = self.ctx
        # if ctx.dim() == 2:
        if not ctx == None:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)
        if not ctxs == None:
            ctxs = ctxs.unsqueeze(0).expand(n_cls, -1, -1)
        # return ctx + embedding,tokenized_prompts

        # prefix = self.token_prefix
        # suffix = self.token_suffix
        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + (self.n_ctx_general+ self.n_ctx_special):, :]

        if self.class_token_position == "end":
            if not ctx == None and not ctxs == None: 
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        ctxs,
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            elif not ctx == None:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            elif not ctxs == None:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctxs,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )


        else:
            raise ValueError

        return prompts, tokenized_prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        # self.dtype = clip_model.dtype
        self.dtype = torch.float32

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return  x

class PromptEnsemble(nn.Module):
    def __init__(self, model):
        super().__init__()
        # self.prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
        # self.prompt_normal = ['flawless {}', 'perfect {}', '{} without flaw', '{} without defect', '{} without damage']
        # self.prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        
        self.prompt_normal = ['normal {}', 'flawless {}', 'perfect {}', '{} without defect']
        self.prompt_abnormal = ['abnormal {}','damaged {}', 'broken {}', '{} with defect']
        self.prompt_state = [self.prompt_normal, self.prompt_abnormal]
        self.promptlearner = PromptLearner(model)
        self.model = model
        
        
    def forward(self, texts):
        text_features = []
        for i in range(len(self.prompt_state)):
            prompted_state = [state.format(texts[0]) for state in self.prompt_state[i]]
            
            learned_prompts, tokenized_prompts = self.promptlearner(prompted_state)
            class_embeddings = TextEncoder(self.model)(learned_prompts, tokenized_prompts)
            #print (class_embeddings.shape)
            #assert 1==2
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=1).t()
        
        return text_features

        
def encode_text_with_prompt_ensemble_ATP(model,promptlearner, texts, device):

    prompt_normal = 'perfect '
    prompt_abnormal = 'damaged '
    prompt_state = [prompt_normal, prompt_abnormal]

    text_prompts = {}
    for text in texts:
        text_features = []
        for i in range(len(prompt_state)):
            # prompted_state = [state.format(text) for state in prompt_state[i]]
            input_texts = prompt_state[i]
            input_texts = input_texts + str(text)
            learned_prompts, tokenized_prompts = promptlearner([input_texts],i)
            class_embeddings = TextEncoder(model)(learned_prompts, tokenized_prompts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=1).to(device).t()
        text_prompts[text] = text_features.permute(1,0)

    return text_prompts  
    
       