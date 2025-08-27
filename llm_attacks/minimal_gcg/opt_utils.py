import gc

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_attacks import get_embedding_matrix, get_embeddings


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    embed_device = embed_weights.device
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=embed_device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids.to(embed_device)[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=embed_device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    input_ids_local = input_ids.to(embed_device)
    embeds = get_embeddings(model, input_ids_local.unsqueeze(0)).detach()
    # Ensure dtype/device match model to avoid bfloat16/float16 mismatch
    model_param = next(model.parameters())
    target_dtype = model_param.dtype
    device = embed_device
    input_embeds = input_embeds.to(dtype=target_dtype, device=device)
    embeds = embeds.to(dtype=target_dtype, device=device)
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids_local[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    
    loss.backward()
    
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    
    return grad


def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)  

    return new_control_toks


def initial_sample_control(control_toks, sample_size, grad, topk=256):
  k = topk
  top_k_indices = (-grad).topk(topk, dim=1).indices
  # Number of samples in the batch
  batch_size = top_k_indices.size(0)
  # sample_size = 512
  # Generate random positions for each sample (values between 0 and k-1)
  random_positions = torch.randint(low=0, high=k, size=(batch_size, sample_size), device=top_k_indices.device)
  # Expand dimensions to match top_k_indices for gathering
  # Shape of random_positions.unsqueeze(1): (batch_size, 1)
  # random_positions_expanded = random_positions.unsqueeze(1)  # Shape: (batch_size, 1)
  # import ipdb; ipdb.set_trace()
  # Gather the randomly selected indices from top_k_indices
  # final_indices will have shape: (batch_size,)
  final_indices = top_k_indices.gather(1, random_positions)
  return final_indices.view(sample_size, -1) 



def half_sample_control(control_toks, sample_size, grad, topk=256, not_allowed_tokens=None):
    """
    Sample control candidates using the same top-k random selection logic as
    initial_sample_control, but for each sample only replace half of the
    positions in control_toks. The other half of positions remain identical
    to the original control_toks.

    Parameters
    ----------
    control_toks : torch.Tensor
        Shape: (seq_len,) token ids of the current control string.
    sample_size : int
        Number of samples to generate.
    grad : torch.Tensor
        Shape: (seq_len, vocab_size) gradients per position.
    topk : int
        Consider top-k candidate tokens per position.
    not_allowed_tokens : torch.Tensor | None
        Optional 1D tensor of token ids to mask out.

    Returns
    -------
    torch.Tensor
        Shape: (sample_size, seq_len) sampled controls where half of positions
        are replaced by top-k random candidates and the rest are kept the same.
    """

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    device = grad.device
    seq_len = control_toks.shape[0]
    control_toks = control_toks.to(device)

    # Top-k indices per position
    top_k_indices = (-grad).topk(topk, dim=1).indices  # (seq_len, topk)

    # Number of positions to replace per sample (roughly half)
    num_replace = max(1, seq_len // 2)

    # Prepare a base tensor filled with the original control tokens for all samples
    samples = control_toks.unsqueeze(0).repeat(sample_size, 1)  # (sample_size, seq_len)

    # For each sample, choose a unique set of positions to replace
    # Build a (sample_size, num_replace) index tensor of positions
    if num_replace == seq_len:
        replace_positions = torch.arange(seq_len, device=device).unsqueeze(0).repeat(sample_size, 1)
    else:
        # Randomly choose num_replace positions without replacement per sample
        replace_positions = []
        arange_positions = torch.arange(seq_len, device=device)
        for _ in range(sample_size):
            perm = torch.randperm(seq_len, device=device)
            replace_positions.append(arange_positions[perm[:num_replace]].unsqueeze(0))
        replace_positions = torch.cat(replace_positions, dim=0)  # (sample_size, num_replace)

    # For each selected position, pick a random token from its top-k candidates
    rand_topk_indices = torch.randint(low=0, high=topk, size=(sample_size, num_replace), device=device)
    # Gather candidate tokens: need to index top_k_indices[position, rand]
    # We'll do position-wise gather using advanced indexing
    gathered_tokens = []
    for i in range(sample_size):
        pos_i = replace_positions[i]  # (num_replace,)
        top_i = top_k_indices[pos_i]  # (num_replace, topk)
        rand_i = rand_topk_indices[i].unsqueeze(1)  # (num_replace, 1)
        token_i = torch.gather(top_i, 1, rand_i).squeeze(1)  # (num_replace,)
        gathered_tokens.append(token_i.unsqueeze(0))
    gathered_tokens = torch.cat(gathered_tokens, dim=0)  # (sample_size, num_replace)

    # Scatter selected tokens back into samples
    samples.scatter_(1, replace_positions, gathered_tokens)

    return samples


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    
    primary_device = get_embedding_matrix(model).device
    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=primary_device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), " 
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(primary_device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(primary_device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits
    

def forward(*, model, input_ids, attention_mask, batch_size=512):

    logits = []
    input_device = get_embedding_matrix(model).device
    for i in range(0, input_ids.shape[0], batch_size):
        
        batch_input_ids = input_ids[i:i+batch_size].to(input_device)
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size].to(input_device)
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask
    
    return torch.cat(logits, dim=0)

def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
    return loss.mean(dim=-1)


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', device_map=None, **kwargs):
    use_auto_map = (isinstance(device, str) and device == 'auto') or (device_map == 'auto')
    if use_auto_map:
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map='auto',
                **kwargs
            ).eval()
    else:
        if isinstance(device, str) and ',' in device:
            raise ValueError("Invalid device string 'cuda:5,6'. Use device_map='auto' or pass a device map dict for multi-GPU placement.")
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                **kwargs
            ).to(device).eval()
    # Normalize model dtype to avoid mixed fp16/bf16 parameters
    try:
        param_dtypes = {p.dtype for p in model.parameters()}
        if len(param_dtypes) > 1:
            # preferred_dtype = torch.bfloat16 if torch.bfloat16 in param_dtypes else torch.float16
            preferred_dtype = torch.float16
            model = model.to(dtype=preferred_dtype)
    except Exception:
        pass

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer