"""Entrypoint for the application."""

import logging
import pathlib

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from app import mcmc, plots

NUM_ITERATIONS = 10000
BURN_IN = 1000
K = 0.1
NUM_COLS_MAIN_CONTENT = 2
NUM_COLS_HISTOGRAMS = 3
PARAM_NAMES = ['a', 'b', 'sigma']
EXAMPLE_IMG_PATH = pathlib.Path("assets") / "synthetic_data.png"

st.set_page_config(layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def initialize_session_state():
    if "running" not in st.session_state:
        st.session_state.running = False
        st.session_state.idx = 0

    if "thetas" not in st.session_state:
        st.session_state.thetas = pd.DataFrame(
            np.zeros((NUM_ITERATIONS, 3)),
            columns=PARAM_NAMES
        )
        st.session_state.thetas.iloc[0] = mcmc.generate_initial_theta()

@st.cache_data
def load_data():
    return np.genfromtxt(
        pathlib.Path("data") / "synthetic_data.csv", 
        delimiter=",", 
        skip_header=True
    )

def start_animation():
    st.session_state.running = True

def stop_animation():
    st.session_state.running = False

def reset_animation():
    st.session_state.idx = 0
    st.session_state.thetas = pd.DataFrame(
        np.zeros((NUM_ITERATIONS, 3)),
        columns=PARAM_NAMES
    )
    st.session_state.thetas.iloc[0] = mcmc.generate_initial_theta()
    LOGGER.info(f"Initial theta after reset: {st.session_state.thetas.iloc[0].values}")

def update_plots(current_data):
    """Update all plots with current data."""
    if not current_data.empty:
        trace_chart = plots.trace_plot(current_data)
        trace_plot_a.altair_chart(trace_chart, use_container_width=True)

    histogram_a = plots.plot_histogram_or_empty(current_data, "a", "Parameter a")
    histogram_b = plots.plot_histogram_or_empty(current_data, "b", "Parameter b")
    hist_sigma = plots.plot_histogram_or_empty(current_data, "sigma", "Parameter σ")

    hist_a.altair_chart(histogram_a, use_container_width=True)
    hist_b.altair_chart(histogram_b, use_container_width=True)
    hist_c.altair_chart(hist_sigma, use_container_width=True)    

def mcmc_animation_plots(data: pd.DataFrame):
    for i in range(st.session_state.idx, NUM_ITERATIONS):
        if not st.session_state.running:
            break
        
        current_theta = st.session_state.thetas.iloc[max(0, i-1)].values
        theta_prime = mcmc.propose_theta(current_theta, K)
        
        proposal_ratio = mcmc.proposal_ratio(current_theta, theta_prime, K)        
        log_posterior_theta_prime = mcmc.log_posterior(data, theta_prime)
        log_posterior_theta = mcmc.log_posterior(data, current_theta)
        
        log_acceptance_ratio = (log_posterior_theta_prime - log_posterior_theta) 
        log_acceptance_ratio += proposal_ratio
        
        if np.log(stats.uniform().rvs()) < log_acceptance_ratio:
            st.session_state.thetas.iloc[i] = theta_prime
        else:
            st.session_state.thetas.iloc[i] = current_theta
        
        st.session_state.idx = i

        if i % 20 == 0 or i == NUM_ITERATIONS - 1:
            update_plots(st.session_state.thetas.iloc[:i+1])

def show_param_summary(thetas: pd.DataFrame):
    st.subheader("Summary of parameters")
    st.markdown("$N_{BurnIn} = %d$" % BURN_IN)
    st.markdown("$N_{samples} = %d$" % NUM_ITERATIONS)
    st.write(thetas.agg(['mean', 'std']))

def show_intro_content():
    """Show background content of the application."""

    st.header("Background")
    col_1, col_2 = st.columns(NUM_COLS_MAIN_CONTENT)

    with col_1:
        st.markdown(
            r"""
            Here we are demonstrating a solution to a a regression problem using 
            Bayesian Analysis.
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
            """)

    with col_2:
        st.image(EXAMPLE_IMG_PATH)

    st.subheader("Proposal Distribution")
    st.markdown(
        r"""
        Our proposal distribution is:

        For convenience we assume that a and b are independent and both normally 
        distributed (iid).
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

        Because the standard deviation is non-negative, we will use the gamma 
        distribution as our prior.
        $$
        \sigma' \sim \Gamma(\sigma k\omega, k\omega)
        $$

        We will use the Metropolis Hastings algorithm to estimate the parameters.
        """)

def show_sidebar_controls():
    """Show sidebar with controls."""
    with st.sidebar:
        st.subheader("Run Animation")
        main_button = st.empty()
        reset_button = st.empty()

        if not st.session_state.running and st.session_state.idx < NUM_ITERATIONS:
            main_button.button("Start", on_click=start_animation)
        else:
            main_button.button("Stop", on_click=stop_animation)

        if not st.session_state.running and st.session_state.idx > 0:
            reset_button.button("Reset", on_click=reset_animation)

initialize_session_state()

st.title("Bayesian Linear Regression with MCMC")
show_sidebar_controls()

# Display plots
st.subheader("MCMC Animation")
with st.container():
    trace_plot_a = st.empty()
    
with st.container():
    col_1, col_2, col3 = st.columns(NUM_COLS_HISTOGRAMS)
    with col_1:
        hist_a = st.empty()
    with col_2:
        hist_b = st.empty()
    with col3:
        hist_c = st.empty()

show_intro_content()

data = load_data()
mcmc_animation_plots(data)

# Discard the initial number of samples as this is when the 
# chain is still settling down
if st.session_state.idx > BURN_IN:
    show_param_summary(st.session_state.thetas[BURN_IN:])

# Update plots with current data, even if animation has stopped
update_plots(st.session_state.thetas.iloc[:st.session_state.idx+1])

