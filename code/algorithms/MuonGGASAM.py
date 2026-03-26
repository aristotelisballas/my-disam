import sys
sys.path.append(sys.path[0].replace('algorithms', ''))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import itertools
from tqdm import tqdm

from algorithms.DISAM import DISAM_Trainer


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the orthogonalization of G.
    Used for spectral (Muon) perturbation.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class MuonGGASAM_Trainer(DISAM_Trainer):
    """
    MuonGGA-SAM Trainer: Combines Spectral Sharpness-Aware Minimization (MuonSAM)
    with Domain-Gradient Similarity Noise (GGA-L).

    Adapted from the DomainBed-style Algorithm class to this repo's trainer pattern.
    Data arrives as merged batches with domain_labels; we split per-domain internally.
    """

    def get_log_name(self):
        base = super().get_log_name()
        return base + f'_rho{self.args.rho}_gamma{self.args.gga_l_gamma}'

    @staticmethod
    def calculate_similarity(grads_list):
        """Average cosine similarity between domain gradients."""
        cos = nn.CosineSimilarity(dim=0)
        if len(grads_list) < 2:
            device = grads_list[0].device if grads_list else torch.device('cpu')
            return torch.tensor(1.0, device=device)
        pairs = list(itertools.combinations(range(len(grads_list)), 2))
        sims = [cos(grads_list[i], grads_list[j]) for i, j in pairs]
        return sum(sims) / len(pairs)

    def muon_gga_sam_train(self, n_epoch, dataloader, model=None, optimizer=None, prefix='merged'):
        if model is None:
            model = self.model
            optimizer = self.optimizer

        model.train()

        for i, data_list in tqdm(enumerate(dataloader)):
            if len(data_list) == 3:
                imgs, labels, domain_labels = data_list
            else:
                imgs, labels = data_list
                domain_labels = torch.zeros(len(labels), dtype=torch.long)

            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            domain_labels = domain_labels.to(self.device)

            if self.data_aug is not None:
                imgs = self.data_aug(imgs)

            # Split batch into per-domain lists
            unique_domains = domain_labels.unique().tolist()
            unique_domains.sort()
            x_list, y_list, chunk_sizes = [], [], []
            for d in unique_domains:
                mask = domain_labels == d
                x_list.append(imgs[mask])
                y_list.append(labels[mask])
                chunk_sizes.append(mask.sum().item())

            num_domains = len(x_list)
            total_samples = sum(chunk_sizes)

            # ==============================================================
            # Phase 1: Single Forward Pass for Domain & Clean Gradients
            # ==============================================================
            all_x = torch.cat(x_list)
            all_y = torch.cat(y_list)
            logits = model(all_x)
            logits_split = torch.split(logits, chunk_sizes)

            domain_grads_flattened = []
            clean_grad_w = [torch.zeros_like(p) for p in model.parameters()]
            loss_clean_val = 0.0

            for di in range(num_domains):
                loss_i = F.cross_entropy(logits_split[di], y_list[di])
                weight_i = chunk_sizes[di] / total_samples
                loss_clean_val += loss_i.item() * weight_i

                retain = (di < num_domains - 1)
                grad_i = autograd.grad(loss_i, model.parameters(), retain_graph=retain)

                domain_grads_flattened.append(
                    torch.cat([g.flatten() for g in grad_i if g is not None])
                )
                for j, g in enumerate(grad_i):
                    if g is not None:
                        clean_grad_w[j] += g.detach() * weight_i

            # ==============================================================
            # Phase 2: Domain Similarity & Noise Scale
            # ==============================================================
            avg_sim = MuonGGASAM_Trainer.calculate_similarity(domain_grads_flattened)
            alpha = self.args.gga_l_gamma * (1.0 - avg_sim)

            # ==============================================================
            # Phase 3: Spectral Perturbations (Muon)
            # ==============================================================
            eps_list = []
            for p, g in zip(model.parameters(), clean_grad_w):
                if g is None:
                    eps_list.append(None)
                    continue
                if g.ndim >= 2:
                    orig_shape = g.shape
                    view_2d = g.view(g.size(0), -1)
                    g_ortho = zeropower_via_newtonschulz5(view_2d)
                    # Instead of: e = g_ortho.view(orig_shape) * self.rho
                    # e = g_ortho.view(orig_shape) * self.args.rho
                    weight_norm = p.norm(2).clamp(min=1e-12) # p is the clean weight tensor
                    e = g_ortho.view(orig_shape) * self.rho * weight_norm
                    eps_list.append(e)
                else:
                    norm_val = g.norm(2)
                    if norm_val > 1e-12:
                        e = (g / norm_val) * self.args.rho
                    else:
                        e = torch.zeros_like(g)
                    eps_list.append(e)

            # ==============================================================
            # Phase 4: Apply Perturbation (Ascent Step)
            # ==============================================================
            with torch.no_grad():
                for p, v in zip(model.parameters(), eps_list):
                    if v is not None:
                        p.add_(v)

            # ==============================================================
            # Phase 5: Adversarial Forward & Backward
            # ==============================================================
            optimizer.zero_grad()
            loss_perturbed = F.cross_entropy(model(all_x), all_y)
            loss_perturbed.backward()

            # ==============================================================
            # Phase 6: Restore Weights & Inject GGA-L Noise
            # ==============================================================
            with torch.no_grad():
                for p, v in zip(model.parameters(), eps_list):
                    if v is not None:
                        p.sub_(v)
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * alpha
                        p.grad.add_(noise)

            # ==============================================================
            # Phase 7: Optimizer Step
            # ==============================================================
            optimizer.step()

            self.metric.update(logits, all_y)

        results_dict = self.metric.results()
        self.log_ten.add_scalar(f'{prefix}_train_loss', results_dict['loss'], n_epoch)

    def run(self):
        self.total_results_dict = {}
        self.best_acc = 0.
        for i in range(self.epochs):
            self.current_epoch = i
            self.total_results_dict[self.current_epoch] = {}
            self.muon_gga_sam_train(i, self.dataloaders_dict['merged']['train'], self.model, self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()

            val_results = self.val(i, 'lodo', self.dataloaders_dict['merged']['val'], self.model, prefix='val')
            self.total_results_dict[self.current_epoch]['val'] = val_results
            is_best = val_results['acc'] > self.best_acc
            self.best_acc = max(val_results['acc'], self.best_acc)
            if is_best:
                self.log_file.info(f'Get Best acc: {self.best_acc*100:.2f}% on epoch {i}')

            test_results = self.val(i, self.test_domain, self.dataloaders_dict[self.test_domain]['test'], self.model, prefix='unseen_test')
            self.total_results_dict[self.current_epoch]['test'] = test_results

            self.save_checkpoint(i, self.model, self.total_results_dict, self.best_acc, self.save_dir, is_best=is_best, prefix='lodo_')
