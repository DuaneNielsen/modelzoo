import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn


class MDNRNN(nn.Module):
    """
    Recurrent mixture density network.

    Description of the network
    """

    def __init__(self, i_size, z_size, hidden_size, num_layers, n_gaussians):
        nn.Module.__init__(self)
        self.a_size = i_size
        self.z_size = z_size
        self.hidden_size = hidden_size
        self.n_gaussians = n_gaussians

        self.lstm = nn.LSTM(i_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)

        self.pi = nn.Linear(hidden_size, z_size * n_gaussians)
        self.lsfm = nn.LogSoftmax(dim=3)
        self.mu = nn.Linear(hidden_size, z_size * n_gaussians)
        self.sigma = nn.Linear(hidden_size, z_size * n_gaussians)

    def forward(self, z, device):
        """
        Compute MDN parameters a mix of gaussians at each timestep.

        z - a list, len(batch_size), of [episode length, latent size]
        pi, mu, sigma - (batch size, episode length, n_gaussians)
        """
        packed = rnn_utils.pack_sequence(z).to(device)
        self.lstm.flatten_parameters()
        packed_output, (hn, cn) = self.lstm(packed)
        output, index = rnn_utils.pad_packed_sequence(packed_output)

        mu, pi, sigma = self.paramaterize_mixture(output)

        return pi, mu, sigma, (hn, cn)

    def paramaterize_mixture(self, output):
        """
        Use the output of the lstm to parameterize a mixture model.

        :param output: the output of the lstm
        :return: mu, pi and sigma, the mean, probability distribution and sigma
        parameters of the mixture model
        """
        episode_length = output.size(0)
        pi = self.pi(output)
        mu = self.mu(output)
        sigma = torch.exp(self.sigma(output))
        pi = pi.view(-1, episode_length, self.z_size, self.n_gaussians)
        mu = mu.view(-1, episode_length, self.z_size, self.n_gaussians)
        sigma = sigma.view(-1, episode_length, self.z_size, self.n_gaussians)
        pi = self.lsfm(pi)
        return mu, pi, sigma

    def step(self, z, context=None):
        """
        Use for auto-regressive generation.

        :param z: observation at t1
        :param context: context at t1
        :return: pi, mu, sigma, context
        """
        if context is None:
            output, context = self.lstm(z)
        else:
            output, context = self.lstm(z, context)
        mu, pi, sigma = self.paramaterize_mixture(output)
        return pi, mu, sigma, context

    def sample(self, pi, mu, sigma):
        """
        Draws a sample from the given parameters.

        :param pi: log prob over distributions
        :param mu: mean
        :param sigma: std_dev
        :return: the output
        """
        prob_pi = torch.exp(pi)
        mn = torch.distributions.multinomial.Multinomial(1, probs=prob_pi)
        mask = mn.sample().byte()
        output_shape = mu.shape[0:-1]
        mu = mu.masked_select(mask).reshape(output_shape)
        sigma = sigma.masked_select(mask).reshape(output_shape)
        mixture = torch.normal(mu, sigma)
        return mixture

    def loss_fn(self, y, pi, mu, sigma):
        """
        Loss function.

        Computes the log probability of the datapoint being
        drawn from all the gaussians parametized by the network.
        Gaussians are weighted according to the pi parameter

        :param y - the target output
        :param pi - log probability over distributions in mixture given x
        :param mu - vector of means of distributions
        :param sigma - vector of standard deviation of distribution
        """
        y = y.unsqueeze(3)
        mixture = torch.distributions.normal.Normal(mu, sigma)
        log_prob = torch.clamp(mixture.log_prob(y), max=0)
        weighted_logprob = log_prob + pi
        log_sum = torch.logsumexp(weighted_logprob, dim=3)
        return -log_sum.mean()
