import torch.nn as nn
import torch
import normflows as nf


class NormalizingFlow(nn.Module):
    """
    Base class for normalizing flow.
    """

    def __init__(self, transforms, input_dim, device="cuda", num_classes=10):
        super(NormalizingFlow, self).__init__()
        self.transforms = transforms  # has to be of type nn.Sequential.

        self.register_buffer('loc', torch.zeros(num_classes, input_dim).to(device))
        self.register_buffer('scale_tril', torch.eye(input_dim).unsqueeze(0).expand(num_classes, -1, -1).to(device))
        self.base_dist = nf.distributions.ClassCondDiagGaussian(512, 10, )

    def forward_kld(self, x, y=None):
        """Estimates forward KL divergence, see see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution
          y: Batch of targets, if applicable

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        return -torch.mean(self.log_prob(x, y))

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

    def log_prob(self, x, y):
        """Calculates the log-likelihood of the given sample x (see equation (1)).

        Args:
            x (torch.Tensor): input
            y : class

        Returns:
            torch.Tensor: log-likelihood of x
        """
        z, log_abs_det = self.forward_and_log_det(x)
        log_pz = self.base_dist.log_prob(z, y)
        log_px = log_pz.to("cuda") + log_abs_det
        return log_px

    def hybrid_loss_gen_part(self, x, num_classes=10):
        batch_size = x.size(0)
        p_x = torch.zeros(batch_size, 1)
        p_xcs = torch.zeros(batch_size, num_classes)
        for c in range(num_classes):
            p_xc_current = torch.exp(self.log_prob(x, torch.tensor(c).expand(batch_size)))
            p_x += p_xc_current.view(-1, 1)
            p_xcs[:, c] = p_xc_current
        return p_x.view(-1), p_xcs

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