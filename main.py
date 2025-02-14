"""Entrypoint for the application."""

import pathlib
import streamlit as st
import numpy as np
import pandas as pd
import logging
from scipy import stats
from app import plots, text, mcmc

# Constants
NUM_ITERATIONS = 10000
K = 0.1

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Initialization
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.idx = 0

if "thetas" not in st.session_state:
    # Initialize thetas as a DataFrame with columns a, b, sigma
    st.session_state.thetas = pd.DataFrame(
        np.zeros((NUM_ITERATIONS, 3)),
        columns=['a', 'b', 'sigma']
    )

@st.cache_data
def load_data():
    """Load data with caching."""
    return np.genfromtxt(
        pathlib.Path("data") / "synthetic_data.csv", 
        delimiter=",", 
        skip_header=True
    )

def start_button():
    st.session_state.running = True

def stop_button():
    st.session_state.running = False

def reset_button():
    st.session_state.idx = 0
    st.session_state.thetas = pd.DataFrame(
        np.zeros((NUM_ITERATIONS, 3)),
        columns=['a', 'b', 'sigma']
    )

def update_plots():
    """Update all plots with current data."""
    current_data = st.session_state.thetas.iloc[:st.session_state.idx+1]
    if not current_data.empty:

        trace_chart = plots.trace_plot(current_data)
        trace_plot_a.altair_chart(trace_chart, use_container_width=True)
        
        # Update histograms
        histogram_a = plots.histogram_plot(current_data, 'a', "Parameter a")
        histogram_b = plots.histogram_plot(current_data, 'b', "Parameter b")
        histogram_sigma = plots.histogram_plot(current_data, 'sigma', "Parameter σ")
        
        hist_a.altair_chart(histogram_a, use_container_width=True)
        hist_b.altair_chart(histogram_b, use_container_width=True)
        hist_sigma.altair_chart(histogram_sigma, use_container_width=True)

def run_animation() -> None:
    """Run the MCMC animation."""
    data = load_data()
    
    # Initialize theta if starting fresh
    if st.session_state.idx == 0:
        initial_values = np.random.rand(3)
        st.session_state.thetas.iloc[0] = initial_values
    
    while st.session_state.idx < NUM_ITERATIONS:
        if not st.session_state.running:
            break

        i = st.session_state.idx
        
        # Get current theta as numpy array for MCMC calculations
        current_theta = st.session_state.thetas.iloc[max(0, i-1)].values
        
        theta_prime = mcmc.propose_theta(current_theta, K)
        
        proposal_ratio = mcmc.proposal_ratio(current_theta, theta_prime, K)
        
        log_posterior_theta_prime = mcmc.log_posterior(data, theta_prime)
        log_posterior_theta = mcmc.log_posterior(data, current_theta)
        
        log_acceptance_ratio = (log_posterior_theta_prime - log_posterior_theta) + proposal_ratio
        
        # Accept or reject the proposal
        if np.log(stats.uniform().rvs(1)) < log_acceptance_ratio:
            st.session_state.thetas.iloc[i] = theta_prime
        else:
            st.session_state.thetas.iloc[i] = current_theta
        
        # Update plots if they're uncommented
        update_plots()
        
        st.session_state.idx += 1

# ---------------- #
#     UI Layout    #
#------------------#
st.title("Bayesian Linear Regression with MCMC")
st.sidebar.title("Inputs")

with st.sidebar:
    main_button = st.empty()
    reset_button_ = st.empty()

    if not st.session_state.running and st.session_state.idx < NUM_ITERATIONS:
        main_button.button("Start", on_click=start_button)
    else:
        main_button.button("Stop", on_click=stop_button)
    
    if not st.session_state.running and st.session_state.idx > 0:
        reset_button_.button("Reset", on_click=reset_button)

# Main content
col_1, col_2 = st.columns(2)

with col_1:
    st.markdown(text.PROBLEM_DESCRIPTION)

with col_2:
    st.image(pathlib.Path("assets") / "synthetic_data.png")

st.markdown(text.PROPOSAL_DISTRIBUTION_TEXT)

# Display plots
with st.container():
    trace_plot_a = st.empty()
    
with st.container():
    col_1, col_2, col3 = st.columns(3)
    with col_1:
        hist_a = st.empty()
    with col_2:
        hist_b = st.empty()
    with col3:
        hist_sigma = st.empty()

# Run simulation
with st.spinner("Running MCMC..."):
    run_animation()

# Update state on animation end
st.session_state.running = False
update_plots()