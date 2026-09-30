"""DeepEcho's PAR with vectorised tensor building, loss and sampling.

Same model (PARNet), same data encoding, same loss, same sampling rules as
`deepecho.models.par.PARModel` (v0.8.1); only the Python loops over sequences and
columns are replaced by tensor operations. Valid when every sequence has the same
length, there is no context and every column is continuous — our setting; otherwise
the original methods are used.

Why it matters: the original loss builds one Normal per (column, sequence) every epoch
and sampling reruns the GRU on the whole prefix for one sequence at a time, so AQI-36
(36 columns × 2000 sequences, length 168) takes hours while the arithmetic is trivial.
Equivalences checked in tests/test_fast_par.py: tensors identical, loss equal to the
original to float precision, incremental GRU state equal to the full-prefix rerun.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from deepecho.models.par import PARModel, PARNet


class FastPARModel(PARModel):
    def _vectorisable(self) -> bool:
        cols = [p for k, p in self._data_map.items() if k != "<TOKEN>"]
        return (self._fixed_length and not self._ctx_dims
                and all(p["type"] == "continuous" for p in cols))

    def _continuous_indices(self):
        keys = sorted(k for k in self._data_map if k != "<TOKEN>")
        idx = np.array([self._data_map[k]["indices"] for k in keys])  # (N, 3): mu, sigma, missing
        mean = np.array([self._data_map[k]["mu"] for k in keys])
        std = np.array([self._data_map[k]["std"] for k in keys])
        return keys, idx, mean, std

    def _sequences_to_tensor(self, sequences) -> torch.Tensor:
        """(L + 2, S, dims): START row, L body rows, END row — as _data_to_tensor per sequence."""
        keys, idx, mean, std = self._continuous_indices()
        data = np.array([[seq["data"][k] for k in keys] for seq in sequences], dtype=np.float64)  # (S, N, L)
        s, n, length = data.shape
        tok = self._data_map["<TOKEN>"]["indices"]
        x = torch.zeros(length + 2, s, self._data_dims)
        x[0, :, tok["<START>"]] = 1.0
        x[-1, :, tok["<END>"]] = 1.0
        body = np.transpose(data, (2, 0, 1))  # (L, S, N)
        missing = np.isnan(body)
        safe_std = np.where(std == 0, 1.0, std)
        mu = np.where(missing | (std == 0), 0.0, (body - mean) / safe_std)
        x[1:-1, :, idx[:, 0]] = torch.as_tensor(mu, dtype=torch.float32)
        x[1:-1, :, idx[:, 2]] = torch.as_tensor(missing, dtype=torch.float32)
        x[1:-1, :, tok["<BODY>"]] = 1.0
        return x.to(self.device)

    def fit_sequences(self, sequences, context_types, data_types):
        self._build(sequences, context_types, data_types)
        if not self._vectorisable():
            return super().fit_sequences(sequences, context_types, data_types)
        x = self._sequences_to_tensor(sequences)
        seq_len = torch.full((x.shape[1],), x.shape[0], dtype=torch.long)
        self._model = PARNet(self._data_dims, self._ctx_dims).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        losses = []
        for epoch in range(self.epochs):
            y = self._model(x, None)
            optimizer.zero_grad()
            loss = self._compute_loss(x[1:], y[:-1], seq_len)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        self.loss_values = pd.DataFrame({"Epoch": np.arange(len(losses)), "Loss": losses})

    def _compute_loss(self, X_padded, Y_padded, seq_len):  # noqa: N803 - library signature
        if not (self._vectorisable() and bool((seq_len == seq_len[0]).all())
                and X_padded.shape[0] <= int(seq_len[0])):
            return super()._compute_loss(X_padded, Y_padded, seq_len)
        _, idx, _, _ = self._continuous_indices()
        batch = X_padded.shape[1]
        mu = Y_padded[:, :, idx[:, 0]]
        sigma = torch.nn.functional.softplus(Y_padded[:, :, idx[:, 1]])
        missing = torch.nn.LogSigmoid()(Y_padded[:, :, idx[:, 2]])
        ll = torch.distributions.normal.Normal(mu, sigma).log_prob(X_padded[:, :, idx[:, 0]]).sum()
        p_true = X_padded[:, :, idx[:, 2]]
        ll = ll + (p_true * missing).sum() + ((1.0 - p_true) * torch.log(1.0 - torch.exp(missing))).sum()
        tok = list(self._data_map["<TOKEN>"]["indices"].values())
        log_softmax = torch.nn.functional.log_softmax(Y_padded[:, :, tok], dim=2)
        target = torch.argmax(X_padded[:, :, tok], dim=2, keepdim=True)
        ll = ll + log_softmax.gather(dim=2, index=target).sum()
        return -ll / (batch * len(self._data_map) * batch)

    @torch.no_grad()
    def sample_batch(self, n: int, length: int, generator: torch.Generator | None = None) -> np.ndarray:
        """n sequences of `length`, all at once; returns (n, length, N) in data units.

        Per step, as _sample_state: Normal(μ, softplus(σ)) for the value, Bernoulli for
        the missing flag (value zeroed when missing), multinomial over the tokens; an END
        before the last step is replaced by BODY, as in _sample_sequence.
        """
        if not self._vectorisable():
            raise ValueError("sample_batch needs fixed-length, context-free, continuous data")
        keys, idx, mean, std = self._continuous_indices()
        tok = self._data_map["<TOKEN>"]["indices"]
        tok_idx = list(tok.values())
        net = self._model
        x = torch.zeros(1, n, self._data_dims, device=self.device)
        x[0, :, tok["<START>"]] = 1.0
        h = None
        rows = []
        for step in range(length):
            out, h = net.rnn(net.down(x), h)
            y = net.up(out)[-1]  # (n, dims)
            nxt = y.clone()
            mu, sigma = y[:, idx[:, 0]], torch.nn.functional.softplus(y[:, idx[:, 1]])
            value = mu + sigma * torch.randn(mu.shape, generator=generator, device=self.device)
            miss = torch.bernoulli(torch.sigmoid(y[:, idx[:, 2]]), generator=generator)
            nxt[:, idx[:, 0]] = value * (1.0 - miss)
            nxt[:, idx[:, 1]] = 0.0
            nxt[:, idx[:, 2]] = miss
            p = torch.softmax(y[:, tok_idx], dim=1)
            choice = torch.multinomial(p, 1, generator=generator)
            onehot = torch.zeros_like(p).scatter_(1, choice, 1.0)
            if step + 1 < length:  # early END is turned into BODY
                end = onehot[:, 1] > 0
                onehot[end, 1], onehot[end, 2] = 0.0, 1.0
            nxt[:, tok_idx] = onehot
            rows.append(nxt[:, idx[:, 0]])
            x = nxt[None]
        values = torch.stack(rows, 1).cpu().numpy()  # (n, length, N), normalised
        return (values * std + mean).astype(np.float32)
