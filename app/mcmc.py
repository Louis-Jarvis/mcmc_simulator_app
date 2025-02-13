import numpy as np
import scipy.stats as stats


NUM_PARAMETERS = 3

# this is an extra parameter that helps to adjust the spread of the distribution
# see here https://medium.com/@tinonucera/bayesian-linear-regression-from-scratch-a-metropolis-hastings-implementation-63526857f191
OMEGA = 500 

def generate_initial_theta():
    theta = np.zeros((3,1))
    theta[:2, 0] = stats.multivariate_normal(0,1).rvs(2)
    theta[2, 0] = stats.gamma(a=1, scale=1).rvs(1)

    return theta.flatten()

# def proposal_ratio(theta: np.ndarray, theta_prime: np.ndarray, k: float) -> np.ndarray:
#     """Likelihood of observing theta-prime given theta"""

#     # what is the likelihood of observing the old parameters given the new parameters
#     # this is the markov chain transition probability q(theta_prime|theta)
#     q_theta_theta_prime = stats.multivariate_normal.logpdf(theta_prime, mean=theta[0], cov=np.eye(2)*k**2)

#     # what is the likelihood of observing the new parameters given the old parameters
#     # this is the markov chain transition probability q(theta|theta_prime)
#     q_theta_prime_theta = stats.multivariate_normal.logpdf(theta, mean=theta_prime[0], cov=np.eye(2)*k**2)

#     # the ratio of the two probabilities is the proposal ratio
#     # we are adding them since we are working with log likelihoods

#     return q_theta_theta_prime - q_theta_prime_theta


def propose_theta(theta: np.ndarray, k: float) -> np.ndarray:
    """
    Propose a new parameter value, theta_prime, given the current parameter vector theta.
    where theta=[a,b,sigma]
    """

    theta_prime = np.zeros(3)

    previous_a_b = theta[0:2]
    previous_sigma = theta[2]
    
    # a and b - proposed based on the previous value of theta
    theta_prime[0:2] = stats.multivariate_normal(mean=previous_a_b, cov=np.eye(2)*k**2).rvs(1)
    
    # sigma
    theta_prime[2] = stats.gamma(a=previous_sigma*k*OMEGA, scale=1/(k*OMEGA)).rvs(1)
    
    return theta_prime

def log_likelihood(data: np.ndarray, theta: np.ndarray) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]    :
    """ 
    Evaluate the likelhood of the data given a specific parameter vector.
    We are assuming the conditional distribution of the response Y|X is normal
    with mean a + b*X and variance sigma^2.
    """

    #TODO
    xs = data[:,0]
    ys = data[:,1]

    a,b,sigma = theta
        
    # we use the log-likeihood so that we are summing instead of multiplying
    # this is to avoid numerical issues with small numbers
    log_lik = stats.norm.logpdf(ys, loc=a + b*xs, scale=sigma)
    return np.sum(log_lik)

def log_prior(theta, sigma=100) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """ 
    Evaluate the log-prior of the parameter vector theta.
    """
    # Similar to the proposal distriibution we will assume a normal prior for a and b
    # as well as a gamma prior for sigma

    a,b,sigma = theta
    # assuming independence we can sum these together
    log_prior = stats.norm.logpdf(a, loc=0, scale=sigma)
    log_prior += stats.norm.logpdf(b, loc=0, scale=sigma) 
    log_prior += stats.gamma.logpdf(sigma, a=1, scale=1)

    return log_prior


def log_posterior(data: np.ndarray, theta: np.ndarray) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    return log_likelihood(data, theta) + log_prior(theta)


def proposal_ratio(theta, theta_prime, k=10):
    """
    this is the proposal distribution ratio
    first, we calculate of the pdf of the proposal distribution at the old value of theta with respect to the new 
    value of theta. And then we do the exact opposite.
    """
    prop_ratio = stats.multivariate_normal.logpdf(theta[:2],mean=theta_prime[:2], cov=np.eye(2)*k**2)
    prop_ratio += stats.gamma.logpdf(theta[2], a=theta_prime[2]*k*500, scale=1/(500*k))
    prop_ratio -= stats.multivariate_normal.logpdf(theta_prime[:2],mean=theta[:2], cov=np.eye(2)*k**2)
    prop_ratio -= stats.gamma.logpdf(theta_prime[2], a=theta[2]*k*500, scale=1/(500*k))
    return prop_ratio
