"""Description text for the application introduction, including latex."""

PROBLEM_DESCRIPTION = r"""
Here we are demonstrating a solution to a a regression problem using Bayesian Analysis.
$$

\hat{Y} = aX + b + \epsilon
$$

We assume:
$$
Y | X \sim N(aX + b, \sigma^2)
$$

We want to estimate the three parameters:
$$
\theta = 
    \begin{pmatrix}
    a \\
    b \\
    \sigma
    \end{pmatrix}
$$
"""

PROPOSAL_DISTRIBUTION_TEXT = r"""
Our proposal distribution is:

For convenience we assume that a and b are independent and both normally distributed (iid).
$$

\begin{pmatrix}
a' \\
b' \\
\end{pmatrix}

= N

\begin{pmatrix}
    \begin{pmatrix}
      a \\
      b \\
\end{pmatrix}
,
\begin{pmatrix}
k^2 & 0 \\
0 & k^2 \\
\end{pmatrix}
\end{pmatrix}
$$

Because the standard deviation is non-negative, we will use the gamma distribution as our prior.
$$
\sigma' \sim \Gamma(\sigma k\omega, k\omega)
$$

We will use the Metropolis Hastings algorithm to estimate the parameters.
"""