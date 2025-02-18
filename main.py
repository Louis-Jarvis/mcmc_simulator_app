"""Entrypoint for the application."""

import pathlib
import logging
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from app import plots, text, mcmc

# Constants
NUM_ITERATIONS = 10000
K = 0.1

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Initialize session state
def initialize_session_state():
    if "running" not in st.session_state:
        st.session_state.running = False
        st.session_state.idx = 0

    if "thetas" not in st.session_state:
        st.session_state.thetas = pd.DataFrame(
            np.zeros((NUM_ITERATIONS, 3)),
            columns=['a', 'b', 'sigma']
        )
        st.session_state.thetas.iloc[0] = mcmc.generate_initial_theta()

# Load data with caching
@st.cache_data
def load_data():
    return np.genfromtxt(
        pathlib.Path("data") / "synthetic_data.csv", 
        delimiter=",", 
        skip_header=True
    )

# Button callbacks
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
    st.session_state.thetas.iloc[0] = mcmc.generate_initial_theta()
    LOGGER.info(f"Initial theta after reset: {st.session_state.thetas.iloc[0].values}")

# Update plots
def update_plots(current_data):
    if not current_data.empty:
        trace_chart = plots.trace_plot(current_data)
        trace_plot_a.altair_chart(trace_chart, use_container_width=True)
        
        histogram_a = plots.histogram_plot(current_data, 'a', "Parameter a")
        histogram_b = plots.histogram_plot(current_data, 'b', "Parameter b")
        histogram_sigma = plots.histogram_plot(current_data, 'sigma', "Parameter σ")
        
        hist_a.altair_chart(histogram_a, use_container_width=True)
        hist_b.altair_chart(histogram_b, use_container_width=True)
        hist_sigma.altair_chart(histogram_sigma, use_container_width=True)

# Run MCMC animation
def run_animation(data: pd.DataFrame):
    for i in range(st.session_state.idx, NUM_ITERATIONS):
        if not st.session_state.running:
            break
        
        current_theta = st.session_state.thetas.iloc[max(0, i-1)].values
        theta_prime = mcmc.propose_theta(current_theta, K)
        
        proposal_ratio = mcmc.proposal_ratio(current_theta, theta_prime, K)        
        log_posterior_theta_prime = mcmc.log_posterior(data, theta_prime)
        log_posterior_theta = mcmc.log_posterior(data, current_theta)
        
        log_acceptance_ratio = (log_posterior_theta_prime - log_posterior_theta) + proposal_ratio
        
        if np.log(stats.uniform().rvs(1)) < log_acceptance_ratio:
            st.session_state.thetas.iloc[i] = theta_prime
        else:
            st.session_state.thetas.iloc[i] = current_theta
        
        st.session_state.idx = i

        if i % 20 == 0 or i == NUM_ITERATIONS - 1:
            update_plots(st.session_state.thetas.iloc[:i+1])

# Initialize session state
initialize_session_state()

# UI Layout
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

# Load data and run simulation
data = load_data()

with st.spinner("Running MCMC..."):
    run_animation(data)

update_plots(st.session_state.thetas.iloc[:st.session_state.idx+1])
