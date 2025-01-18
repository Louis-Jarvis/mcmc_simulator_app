import streamlit as st
import numpy as np
import seaborn as sns
import logging
import pandas as pd

import pathlib

from src.text import PROBLEM_DESCRIPTION, PROPOSAL_DISTRIBUTION_TEXT

NUM_ITERATIONS = 1000

logging.basicConfig(level=logging.INFO)  # TODO change
logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid")

# Generate data
x = np.linspace(0, 10, NUM_ITERATIONS)
y = np.sin(x) + np.random.normal(0, 0.1, NUM_ITERATIONS)

# Initialize the trace plot with an empty DataFrame
if "trace_data" not in st.session_state:
    st.session_state.trace_data = pd.DataFrame(columns=['x', 'y'])

if "hist_data" not in st.session_state:
    st.session_state.hist_data = pd.DataFrame(columns=['Bins', 'Counts'])

def update_trace_plot(x, y, idx):
    # Append new data to the existing DataFrame
    new_data = pd.DataFrame({'x': x[:idx], 'y': y[:idx]})
    st.session_state.trace_data = pd.concat([st.session_state.trace_data, new_data], ignore_index=True)

def update_hist_plot(y, idx):
    # Create a histogram and convert it to a DataFrame for bar chart
    counts, bin_edges = np.histogram(y[:idx], bins=10)  # Adjust the number of bins as needed
    hist_data = pd.DataFrame({'Bins': bin_edges[:-1], 'Counts': counts})
    st.session_state.hist_data = hist_data

# Initialization
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.idx = 0

# Sidebar controls
st.title("Bayesian Linear Regression with MCMC")
st.sidebar.title("Inputs")

with st.sidebar:
    main_button = st.empty()
    reset_button_ = st.empty()

    if not st.session_state.running:
        logger.debug("Starting")
        main_button.button("Start", on_click=lambda: setattr(st.session_state, 'running', True))
    else:
        main_button.button("Stop", on_click=lambda: setattr(st.session_state, 'running', False))
    
    if st.session_state.running or st.session_state.idx > 0:
        logger.debug("Reset button")
        reset_button_.button("Reset", on_click=lambda: setattr(st.session_state, 'idx', 0))

# Main content
col1, col2 = st.columns(2)

with col1:
    st.markdown(PROBLEM_DESCRIPTION)

with col2:
    st.image(pathlib.Path("assets") / "synthetic_data.png")

st.markdown(PROPOSAL_DISTRIBUTION_TEXT)

# Run the simulation and display the plots
with st.container():
    trace_plot_a = st.empty()

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        bar_chart_a = st.empty()
    with col2:
        bar_chart_b = st.empty()
    with col3:
        bar_chart_c = st.empty()

with st.spinner("Running MCMC..."):
    for i in range(0, NUM_ITERATIONS, 10):
        if not st.session_state.running:
            break

        logger.debug(f"Iteration {i}")

        # Update the trace plot data
        update_trace_plot(x, y, i)

        # Use st.line_chart to display the trace plot
        trace_plot_a.line_chart(st.session_state.trace_data.set_index('x'))

        # Update histogram data
        update_hist_plot(y, i)

        # Use st.bar_chart to display the histogram as a bar chart
        bar_chart_a.bar_chart(st.session_state.hist_data.set_index('Bins'))
        bar_chart_b.bar_chart(st.session_state.hist_data.set_index('Bins'))
        bar_chart_c.bar_chart(st.session_state.hist_data.set_index('Bins')) 

        st.session_state.idx += 1

