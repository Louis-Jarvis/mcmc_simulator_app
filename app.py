import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

import pathlib

from src.text import PROBLEM_DESCRIPTION, PROPOSAL_DISTRIBUTION_TEXT

NUM_ITERATIONS = 100

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid")

def generate_trace_plots(x, y, idx, element):
    trace_fig, trace_ax = plt.subplots()
    sns.lineplot(x=x[:idx], y=y[:idx], ax=trace_ax)
    element.pyplot(trace_fig)

def generate_hist_plots(y, idx, element):
    fig, ax = plt.subplots()
    sns.histplot(y[:idx], ax=ax)
    element.pyplot(fig)

# def reset_plots(trace_plots, hist_plots):
#     for plot in trace_plots:
#         plot.empty()
#     for plot in hist_plots:
#         plot.empty()

#TODO eventually get rid of this.
x = np.linspace(0, 10, NUM_ITERATIONS)
y = np.sin(x) + np.random.normal(0, 0.1, NUM_ITERATIONS)

if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.idx = 0


#TODO add dependency injection
def start_button():
    st.session_state.running = True

def stop_button():
    st.session_state.running = False

def reset_button():
    st.session_state.running = False
    st.session_state.idx = 0


st.title("Bayesian Linear Regression with MCMC")

#############################################
# Define the sidebar to control the animation
#############################################
st.sidebar.title("Inputs")

with st.sidebar:

    main_button = st.empty()
    reset_button_ = st.empty()

    if not st.session_state.running:
        main_button.button("Start", on_click=start_button)
    else:
        main_button.button("Stop", on_click=stop_button)
    
    if st.session_state.running or st.session_state.idx > 0:
        logger.debug("Reset button")
        reset_button_.button("Reset", on_click=reset_button)


##############################################
# Define the main content
##############################################  

#TODO tidy all this
col1, col2 = st.columns(2)

with col1:
    st.markdown(PROBLEM_DESCRIPTION)

with col2:
    st.image(pathlib.Path("assets") / "synthetic_data.png")

st.markdown(PROPOSAL_DISTRIBUTION_TEXT)

col1, col2, col3 = st.columns(3)

with col1:
    #TODO turn each one of these columns into a function
    trace_plot_a = st.empty()
    hist_plot_a = st.empty()


with col2:
    trace_plot_b = st.empty()
    hist_plot_b = st.empty()
with col3:
    trace_plot_c = st.empty()
    hist_plot_c = st.empty()

# TODO parallelise
if st.session_state.idx > 0:

    generate_trace_plots(x, y, st.session_state.idx, trace_plot_a)
    generate_trace_plots(x, y, st.session_state.idx, trace_plot_b)
    generate_trace_plots(x, y, st.session_state.idx, trace_plot_c)
    
    generate_hist_plots(y, st.session_state.idx, hist_plot_a)
    generate_hist_plots(y, st.session_state.idx, hist_plot_b)
    generate_hist_plots(y, st.session_state.idx, hist_plot_c)

#TODO run the animation here
#TODO use this in parallel
#TODO refactor this to be a class

with st.spinner("Running MCMC..."):

    while st.session_state.running and st.session_state.idx < NUM_ITERATIONS:
        
        generate_trace_plots(x, y, st.session_state.idx, trace_plot_a)
        generate_trace_plots(x, y, st.session_state.idx, trace_plot_b)
        generate_trace_plots(x, y, st.session_state.idx, trace_plot_c)

        generate_hist_plots(y, st.session_state.idx, hist_plot_a)
        generate_hist_plots(y, st.session_state.idx, hist_plot_b)
        generate_hist_plots(y, st.session_state.idx, hist_plot_c)
        
        st.session_state.idx += 1


