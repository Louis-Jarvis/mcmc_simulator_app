"""Entrypoint for the application."""

import streamlit as st
import numpy as np
import pandas as pd
import logging
import pathlib

from app.plots import trace_plot, histogram_plot
from app.text import PROBLEM_DESCRIPTION, PROPOSAL_DISTRIBUTION_TEXT

NUM_ITERATIONS = 1000

# Generate data
x = np.linspace(0, 10, NUM_ITERATIONS)
y = np.sin(x) + np.random.normal(0, 0.1, NUM_ITERATIONS)

logging.basicConfig(level=logging.INFO)  # TODO change
logger = logging.getLogger(__name__)

# Initialization
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.idx = 0

if "trace_data" not in st.session_state:
    st.session_state.trace_data = pd.DataFrame({"x": [], "y": []})

def start_button():
    st.session_state.running = True

def stop_button():
    st.session_state.running = False

def reset_button():
    st.session_state.idx = 0
    st.session_state.trace_data = pd.DataFrame({"x": [], "y": []})

# Sidebar controls
st.title("Bayesian Linear Regression with MCMC")
st.sidebar.title("Inputs")

with st.sidebar:
    main_button = st.empty()
    reset_button_ = st.empty()

    if not st.session_state.running and st.session_state.idx < NUM_ITERATIONS:
        logger.debug("Starting")
        main_button.button("Start", on_click=start_button)
    else:
        main_button.button("Stop", on_click=stop_button)
    
    if not st.session_state.running and st.session_state.idx > 0:
        logger.debug("Reset button")
        reset_button_.button("Reset", on_click=reset_button)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.markdown(PROBLEM_DESCRIPTION)

with col2:
    st.image(pathlib.Path("assets") / "synthetic_data.png")

st.markdown(PROPOSAL_DISTRIBUTION_TEXT)

# Run the simulation and display the plots
with st.container():
    trace_plot_a = st.altair_chart(trace_plot(st.session_state.trace_data), use_container_width=True)
    
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        bar_chart_a = st.altair_chart(histogram_plot(st.session_state.trace_data, 'y', "y"), use_container_width=True)
    with col2:
        bar_chart_b = st.altair_chart(histogram_plot(st.session_state.trace_data, 'y', "y"), use_container_width=True)
    with col3:
        bar_chart_c = st.altair_chart(histogram_plot(st.session_state.trace_data, 'y', "y"), use_container_width=True)

# Logic for the simulation
with st.spinner("Running MCMC..."):

    while st.session_state.idx < NUM_ITERATIONS:
        if not st.session_state.running:
            break

        logger.debug(f"Iteration {st.session_state.idx}")

        st.session_state.trace_data = pd.concat([
            st.session_state.trace_data, 
            pd.DataFrame(data={
                "x": x[st.session_state.idx], 
                "y": y[st.session_state.idx]
                }, 
                index=[st.session_state.idx])],
                ignore_index=True)
        
        trace_chart = trace_plot(st.session_state.trace_data)
        trace_plot_a.altair_chart(trace_chart, use_container_width=True)

        #TODO replace this with the actual plots
        # # Create Altair histogram styled like Seaborn with no gaps between bars
        histogram = histogram_plot(st.session_state.trace_data, 'y', "y")

        # # Display the Altair histogram
        bar_chart_a.altair_chart(histogram, use_container_width=True)
        bar_chart_b.altair_chart(histogram, use_container_width=True)
        bar_chart_c.altair_chart(histogram, use_container_width=True)



        st.session_state.idx += 1

# on animation end
st.session_state.running = False

