"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from blipnet.losses import GenericLoss


class ContrastiveFragmentLoss(GenericLoss):
    """
    An InfoNCE loss designed to separate nodes in embedding space.
    """
    def __init__(
        self,
        name: str = 'contrastive_fragment_loss',
        alpha: float = 0.0,
        temperature: float = 0.1,
        target: str = '',
        output: str = '',
        meta: dict = {}
    ):
        super(ContrastiveFragmentLoss, self).__init__(
            name, alpha, meta
        )
        self.target = target
        self.output = output
        self.temperature = temperature

    def _loss(
        self,
        data
    ):
        fragment_loss_answer = data[self.target].to(self.device).float().unsqueeze(1)
        fragment_loss_output = data[self.output].to(self.device)

        # Normalize edge embeddings for cosine similarity
        fragment_loss_output = F.normalize(fragment_loss_output, dim=1)

        # Compute similarity matrix (E x E)
        similarity_matrix = torch.mm(
            fragment_loss_output, fragment_loss_output.T
        ) / self.temperature

        positive_mask = (fragment_loss_answer == fragment_loss_answer.T).float()  # (N, N), 1 for positive pairs, 0 for others

        # Remove self-similarities (diagonal entries)
        diag_mask = torch.eye(fragment_loss_output.size(0), device=fragment_loss_output.device)
        positive_mask -= diag_mask  # (N, N), set diagonal to 0

        # Compute log of softmax denominator (all similarities)
        log_sum_exp = torch.logsumexp(similarity_matrix, dim=1)  # (N,)

        # Compute numerator (only positive similarities)
        positive_sim = similarity_matrix * positive_mask  # Mask out non-positive pairs
        log_positive_sim = torch.log(positive_sim.sum(dim=1) + 1e-9)  # Avoid log(0)

        # Compute InfoNCE loss
        infonce_loss = (-log_positive_sim + log_sum_exp).mean()
        data['contrastive_fragment_loss'] = self.alpha * infonce_loss
        return data
