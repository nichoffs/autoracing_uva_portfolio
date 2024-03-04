import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    """
    (B,T,C)->(B,T,C)

    Modified implementation of Vector Quantizer from https://juliusruseckas.github.io/ml/vq-vae.html
    I choose (B,T,C) input instead of (B,C,T).
    Instead of computing losses inside forward function, I leave this to a different place. Why do it there?

    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.reset_parameters()

    def forward(self, latents):
        # Compute L2 distances between latents and embedding weights

        dist = torch.linalg.vector_norm(
            latents.unsqueeze(-2) - self.embedding.weight, dim=-1
        )
        encoding_inds = torch.argmin(
            dist, dim=-1
        )  # Get the number of the nearest codebook vector
        quantized_latents = self.quantize(encoding_inds)  # Quantize the latents
        # Make the gradient with respect to latents be equal to the gradient with respect to quantized latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents

    def quantize(self, encoding_indices):
        z = self.embedding(encoding_indices)
        return z

    def reset_parameters(self):
        nn.init.uniform_(
            self.embedding.weight, -1 / self.num_embeddings, 1 / self.num_embeddings
        )
