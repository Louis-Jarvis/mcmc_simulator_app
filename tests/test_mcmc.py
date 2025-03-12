import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis import given
from app import mcmc

@st.composite
def theta(draw):
    theta = draw(
        arrays(
            dtype=np.float64,
            shape=(3,),
            elements=st.floats(
                min_value=1,  # Small positive number to avoid zero
                max_value=100,    # Adjust as necessary
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    return theta

@st.composite
def data(draw):
    return draw(
        arrays(
            dtype=np.float64, 
            shape=(10, 2), 
            elements=st.floats(
                min_value=1, 
                max_value=100, 
                allow_nan=False, 
                allow_infinity=False
            )
        )
    )

@given(theta=theta(), k=st.floats(min_value=0.01, max_value=1.0))
def test_propose_theta_gives_new_theta(theta, k):
    theta_prime = mcmc.propose_theta(theta, k)
    assert not np.array_equal(theta, theta_prime), "propose_theta did not modify theta as expected."

@given(theta=theta(), data=data())
def test_log_likelihood_is_scalar(theta, data):
    log_lik = mcmc.log_likelihood(data, theta)
    assert log_lik.shape == ()
    assert isinstance(log_lik, np.float64)

@given(theta=theta())
def test_log_prior_is_scalar(theta):
    prior = mcmc.log_prior(theta)
    assert prior.shape == ()
    assert isinstance(prior, np.float64)

@given(theta=theta(), data=data())
def test_log_posterior_is_scalar(theta, data):
    log_posterior = mcmc.log_posterior(data, theta)
    assert log_posterior.shape == ()
    assert isinstance(log_posterior, np.float64)
