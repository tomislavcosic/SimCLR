import torch.nn as nn
import torch


class NormalizingFlow(nn.Module):
    """
    Base class for normalizing flow.
    """

    def __init__(self, transforms, input_dim, device="cuda"):
        super(NormalizingFlow, self).__init__()
        self.transforms = transforms  # has to be of type nn.Sequential.

        self.register_buffer('loc', torch.zeros(input_dim).to(device))
        self.register_buffer('log_scale', torch.zeros(input_dim).to(device))
        self.base_dist = torch.distributions.Normal(self.loc, torch.exp(self.log_scale))

    def forward_and_log_det(self, x):
        """Transforms the input sample to the latent representation z.

        Args:
            x (torch.Tensor): input sample

        Returns:
            torch.Tensor: latent representation of the input sample
        """
        sum_log_abs_det = torch.zeros(len(x), device=x.device)
        z = x
        for transform in self.transforms:
            z, log_abs_det = transform(z)
            sum_log_abs_det += log_abs_det

        return z, sum_log_abs_det

    def forward(self, x):
        """Transforms the input sample to the latent representation z.

        Args:
            x (torch.Tensor): input sample

        Returns:
            torch.Tensor: latent representation of the input sample
        """
        z = x
        for transform in self.transforms:
            z, _ = transform(z)

        return z

    def inverse(self, z):
        """Transforms the latent representation z back to the input space.

        Args:
            z (torch.Tensor): latent representation

        Returns:
            torch.Tensor: representation in the input space
        """
        x = z
        for transform in self.transforms[::-1]:
            x = transform.inverse(x)
        return x

    def log_prob(self, x):
        """Calculates the log-likelihood of the given sample x (see equation (1)).

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: log-likelihood of x
        """
        z, log_abs_det = self.forward_and_log_det(x)
        log_pz = torch.sum(self.base_dist.log_prob(z), axis=1)
        log_px = log_pz.to("cuda") + log_abs_det
        return log_px

    def sample(self, num_samples, T=1):
        """Generates new samples from the normalizing flow.

        Args:
            num_samples (int): number of samples to generate
            T (float, optional): sampling temperature. Defaults to 1.

        Returns:
            torch.Tensor: generated samples
        """

        z = self.base_dist.sample(torch.Size([num_samples])) * T
        x = self.inverse(z)
        return x