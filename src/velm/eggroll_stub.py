import math
import copy
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import torch
import torch.nn as nn
import numpy as np

# Try to import pycma for a full CMA-ES implementation; optional
try:
    import cma
    HAS_PYCMA = True
except Exception:
    HAS_PYCMA = False


class Eggroll:
    """Evolutionary Strategy (ES) trainer with optional parallel evaluation and
    support for evolving adapter-only or all trainable parameters.

    Features:
      - mirrored-sampling ES with normalized rewards
      - optional threaded or process-based parallel evaluation
      - option to evolve only 'adapter' params or 'all' trainable params
      - optional pycma-based CMA-ES when pycma is installed

    API: run_es(model, fitness_fn, pop_size=16, generations=5, sigma=0.02,
                lr=0.1, device=None, num_workers=1, evolve='adapter', algorithm='es', use_processes=False)

    Note: process-based parallel evaluation requires that `model` and `fitness_fn`
    are picklable. If not picklable, the code will automatically fall back to
    threaded evaluation.
    """

    def __init__(self, rank=None, sigma=2e-2):
        self.rank = rank
        self.sigma = sigma

    def _collect_params(self, model, evolve='adapter'):
        params = []
        names = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if evolve == 'adapter':
                if 'adapter' in n:
                    names.append(n)
                    params.append(p)
            else:
                # evolve == 'all'
                names.append(n)
                params.append(p)
        return names, params

    def _get_flat_vector(self, params):
        views = []
        shapes = []
        for p in params:
            shapes.append(p.size())
            views.append(p.detach().reshape(-1))
        if len(views) == 0:
            return torch.tensor([], device='cpu'), shapes
        return torch.cat(views).clone(), shapes

    def _set_from_vector(self, params, shapes, vec):
        offset = 0
        for p, s in zip(params, shapes):
            nels = int(torch.tensor(list(s)).prod().item())
            chunk = vec[offset:offset+nels].view(s)
            p.data.copy_(chunk)
            offset += nels

    # helper used by ThreadPoolExecutor
    def _eval_candidate(self, model_template, candidate_vec, shapes, device, fitness_fn):
        model_copy = copy.deepcopy(model_template)
        model_copy.to(device)
        names, params = self._collect_params(model_copy, evolve='all' if True else 'adapter')
        self._set_from_vector(params, shapes, candidate_vec)
        try:
            fit = float(fitness_fn(model_copy))
        except Exception:
            fit = -1e9
        return fit

    # module-level worker used by multiprocessing Pool - must be picklable reference
    @staticmethod
    def _mp_worker(args):
        # args: (model_pickle, fitness_pickle, candidate_np, shapes, device)
        model_pickle, fitness_pickle, candidate_np, shapes, device = args
        model = pickle.loads(model_pickle)
        fitness_fn = pickle.loads(fitness_pickle)
        model.to(device)
        # collect trainable params in the model (all or adapter determined by shapes length)
        params = [p for n, p in model.named_parameters() if p.requires_grad]
        # set flat vector
        vec = torch.from_numpy(candidate_np).to(next(model.parameters()).device)
        offset = 0
        for p, s in zip(params, shapes):
            nels = int(torch.tensor(list(s)).prod().item())
            chunk = vec[offset:offset+nels].view(s)
            p.data.copy_(chunk)
            offset += nels
        try:
            fit = float(fitness_fn(model))
        except Exception:
            fit = -1e9
        return fit

    def run_es(self, model, fitness_fn, pop_size=16, generations=5, sigma=None, lr=0.1, device=None, num_workers=1, evolve='adapter', algorithm='es', use_processes=False):
        """Run ES/CMA-like optimization and update selected parameters in-place.

        If use_processes=True, attempt to use multiprocessing.Pool for parallel eval.
        This requires that `model` and `fitness_fn` are picklable; otherwise the
        function falls back to threaded evaluation.
        """
        if sigma is None:
            sigma = self.sigma
        if pop_size % 2 == 1:
            pop_size += 1
        names, params = self._collect_params(model, evolve=evolve)
        if len(params) == 0:
            raise RuntimeError('No parameters found to evolve')
        base_vec, shapes = self._get_flat_vector(params)
        dim = base_vec.numel()
        base_vec = base_vec.to(device if device is not None else base_vec.device)

        device = device if device is not None else base_vec.device

        # If pycma available and requested, run a small CMA run (numpy-based)
        if algorithm == 'cma' and HAS_PYCMA:
            mean = base_vec.cpu().numpy()
            sigma0 = float(sigma)
            es = cma.CMAEvolutionStrategy(mean, sigma0, {'popsize': pop_size})
            for g in range(generations):
                solutions = es.ask()
                rewards = []
                for s in solutions:
                    vec = torch.tensor(s, dtype=base_vec.dtype, device=base_vec.device)
                    self._set_from_vector(params, shapes, vec)
                    try:
                        fit = float(fitness_fn(model))
                    except Exception:
                        fit = -1e9
                    rewards.append(-fit)  # CMA minimizes
                es.tell(solutions, rewards)
            best = es.result.xbest
            best_vec = torch.tensor(best, dtype=base_vec.dtype, device=base_vec.device)
            self._set_from_vector(params, shapes, best_vec)
            return True

        for gen in range(generations):
            noises = torch.randn(pop_size, dim, device=base_vec.device)
            candidates = [base_vec + sigma * noises[i] for i in range(pop_size)]

            # PROCESS-BASED parallelism if requested and pickling permitted
            if use_processes and num_workers and num_workers > 1:
                try:
                    model_pickle = pickle.dumps(model)
                    fitness_pickle = pickle.dumps(fitness_fn)
                    # prepare args list
                    args_list = []
                    for i in range(pop_size):
                        cand_np = candidates[i].cpu().numpy()
                        args_list.append((model_pickle, fitness_pickle, cand_np, shapes, str(device)))
                    ctx = mp.get_context('fork')
                    with ctx.Pool(processes=num_workers) as pool:
                        results = pool.map(Eggroll._mp_worker, args_list)
                    rewards = torch.tensor(results, device=base_vec.device)
                except Exception:
                    # fallback to threaded evaluation if pickling fails
                    rewards = None
            else:
                rewards = None

            # If rewards not computed via processes, use threaded or single-thread
            if rewards is None:
                if num_workers is None or num_workers <= 1:
                    rewards = []
                    for i in range(pop_size):
                        self._set_from_vector(params, shapes, candidates[i])
                        try:
                            fit = float(fitness_fn(model))
                        except Exception:
                            fit = -1e9
                        rewards.append(fit)
                    rewards = torch.tensor(rewards, device=base_vec.device)
                else:
                    with ThreadPoolExecutor(max_workers=num_workers) as ex:
                        futures = [ex.submit(self._eval_candidate, model, candidates[i].cpu(), shapes, device, fitness_fn) for i in range(pop_size)]
                        results = [f.result() for f in futures]
                    rewards = torch.tensor(results, device=base_vec.device)

            mean = rewards.mean()
            std = rewards.std()
            if std.item() < 1e-8:
                normalized = rewards - mean
            else:
                normalized = (rewards - mean) / (std + 1e-8)

            if algorithm == 'es':
                update = (normalized.unsqueeze(1) * noises).mean(dim=0)
                base_vec = base_vec + (lr / (sigma)) * update
            elif algorithm == 'cma':
                # fallback lightweight cma-like top-k update
                k = max(1, pop_size // 4)
                topk = torch.topk(rewards, k=k)
                weights = torch.softmax(topk.values, dim=0)
                idxs = topk.indices
                selected_noises = noises[idxs]
                weighted = (weights.unsqueeze(1) * selected_noises).sum(dim=0)
                base_vec = base_vec + (lr / (sigma)) * weighted
                if rewards.mean().item() > mean.item():
                    sigma *= 0.99
            else:
                raise ValueError('Unknown algorithm: ' + str(algorithm))

        self._set_from_vector(params, shapes, base_vec)
        return True
