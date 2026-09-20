"""Additive steering at one decoder block, with a coefficient for every token position.

The hook adds `coeff[b, t] * v` to the block's output hidden state. A history is replayed by
passing the whole token sequence with its coefficient schedule; the same schedule can also be
fed in pieces against a KV cache, and the two must agree (tests/test_steering.py).
"""

import torch


def decoder_layers(model):
    """The decoder's block list, with or without a peft wrapper around the model."""
    for m in model.modules():
        if isinstance(getattr(m, "layers", None), torch.nn.ModuleList):
            return m.layers
    raise ValueError("no decoder block list found")


class Steerer:
    def __init__(self, model, vector, layer, monitor_layer):
        self.model = model
        self.layers = decoder_layers(model)
        dev = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.v = vector.to(dev, dtype=dtype)
        self.unit = (vector.float() / vector.float().norm()).to(dev)
        self.layer, self.monitor_layer = layer, monitor_layer
        self.coeff = None
        self.trace = {}
        self._handles = [
            self.layers[layer].register_forward_hook(self._steer),
            self.layers[monitor_layer].register_forward_hook(self._monitor),
        ]

    def close(self):
        for h in self._handles:
            h.remove()

    def _steer(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        self.trace["pre"] = (hs.float() @ self.unit).detach()
        if self.coeff is not None and bool((self.coeff != 0).any()):
            hs = hs + self.coeff.to(hs.dtype)[:, :, None] * self.v
        self.trace["post"] = (hs.float() @ self.unit).detach()
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    def _monitor(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        self.trace["monitor"] = (hs.float() @ self.unit).detach()
        return output

    @torch.inference_mode()
    def forward(self, ids, coeff, past=None, attention_mask=None, position_ids=None, keep=1):
        """One pass over `ids` [B, T] with `coeff` [B, T]. Returns logits, cache, projections.

        Logits are for the last position only unless `keep` is 0, which keeps every position.
        """
        assert ids.shape == coeff.shape, (ids.shape, coeff.shape)
        self.coeff = coeff.to(self.v.device, dtype=torch.float32)
        try:
            out = self.model(input_ids=ids, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past, use_cache=True, logits_to_keep=keep)  # fmt: skip
        finally:
            self.coeff = None
        trace = {k: v.cpu() for k, v in self.trace.items()}
        logits = out.logits.float() if keep == 0 else out.logits[:, -1, :].float()
        return logits, out.past_key_values, trace
