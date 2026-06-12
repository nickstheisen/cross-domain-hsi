#!/usr/bin/env python

import torch

def predict_batch(model, batch, device='cuda', sub_batch_size=None):
    model.to(device)
    model.eval()

    inputs, _ = batch
    inputs.to(device)
    if sub_batch_size is None:
        preds = model(inputs.to(device))
    else:
        preds = []
        for i in range(0, inputs.shape[0], sub_batch_size):
            sub_inputs = inputs[i:i+sub_batch_size]
            sub_preds = model(sub_inputs.to(device))
            preds.append(sub_preds.detach().cpu())
        preds = torch.vstack(preds)
    return preds

