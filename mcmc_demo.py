"""

"""
import pathlib
import logging

from app import mcmc

import numpy as np

import scipy.stats as stats

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

#np.random.seed(42)

K = 3
NUM_ITERATIONS = 100

def metropolis_hastings(n):

    acceptance_count = 0

    thetas = np.zeros((3, NUM_ITERATIONS))
    thetas[:,0] = mcmc.generate_initial_theta()

    data = np.genfromtxt(
        pathlib.Path("data") / "synthetic_data.csv", 
        delimiter=",", 
        skip_header=True
    )

    for i in range(1, n):

        LOGGER.debug("Iteration %d", i)
        theta_prime = mcmc.propose_theta(thetas[:,i-1], 0.5)

        proposal_ratio = mcmc.proposal_ratio(thetas[:, i-1], theta_prime, 10)

        log_posterior_theta_prime = mcmc.log_posterior(data, theta_prime)
        log_posterior_theta = mcmc.log_posterior(data, thetas[:, i-1])
        
        # instead of dividing we are subtracting the log values
        log_acceptance_ratio = (log_posterior_theta_prime - log_posterior_theta) + proposal_ratio
        
        # exponentiate the log acceptance ratio to obtain probability
        if np.log(stats.uniform().rvs(1)) < log_acceptance_ratio:

            acceptance_count += 1
            LOGGER.debug("Acceptance ratio: %f", log_acceptance_ratio)
            LOGGER.debug(f"Accept theta_prime {theta_prime}")
            thetas[:,i] = theta_prime
        else:
            thetas[:,i] = thetas[:,i-1]
    
    LOGGER.debug("Acceptance rate: %f", acceptance_count/n)
    return thetas

def run_demo():
    thetas = metropolis_hastings(NUM_ITERATIONS)
    print(thetas[-1,:])

if __name__=="__main__":
    run_demo()