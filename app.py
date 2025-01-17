import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import logging

NUM_ITERATIONS = 100

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid")

#TODO eventually get rid of this.
x = np.linspace(0, 10, NUM_ITERATIONS)
y = np.sin(x) + np.random.normal(0, 0.1, NUM_ITERATIONS)

if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.idx = 0


def _start_button():
    st.session_state.running = True

def _stop_button():
    st.session_state.running = False

def _reset_button():
    st.session_state.running = False
    st.session_state.idx = 0


st.title("Bayesian Linear Regression with MCMC")

#############################################
# Define the sidebar to control the animation
#############################################
st.sidebar.title("Inputs")

with st.sidebar:

    main_button = st.empty()
    reset_button = st.empty()

    if not st.session_state.running:
        main_button.button("Start", on_click=_start_button)
    else:
        main_button.button("Stop", on_click=_stop_button)
    
    if st.session_state.running or st.session_state.idx > 0:
        logger.debug("Reset button")
        reset_button.button("Reset", on_click=_reset_button)


##############################################
# Define the main content
##############################################  

st.markdown(
"""

$$
\hat{Y} = aX + b + \epsilon
$$

$$
Y | X \sim N(aX + b, \sigma^2)
$$
""")

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

if st.session_state.idx > 0:
    
    trace_fig_a, trace_ax_a = plt.subplots()
    sns.lineplot(x=x[:st.session_state.idx], y=y[:st.session_state.idx], ax=trace_ax_a)
    trace_plot_a.pyplot(trace_fig_a)
    
    trace_fig_b, trace_ax_b = plt.subplots()
    sns.lineplot(x=x[:st.session_state.idx], y=y[:st.session_state.idx], ax=trace_ax_b)
    trace_plot_b.pyplot(trace_fig_b)

    trace_fig_c, trace_ax_c = plt.subplots()
    sns.lineplot(x=x[:st.session_state.idx], y=y[:st.session_state.idx], ax=trace_ax_c)
    trace_plot_c.pyplot(trace_fig_c)

    #TODO make this a function
    fig_a, ax_a = plt.subplots()
    sns.histplot(y[:st.session_state.idx], ax=ax_a)
    hist_plot_a.pyplot(fig_a)

    fig_b, ax_b = plt.subplots()
    sns.histplot(y[:st.session_state.idx], ax=ax_b)
    hist_plot_b.pyplot(fig_b)

    fig_c, ax_c = plt.subplots()
    sns.histplot(y[:st.session_state.idx], ax=ax_c)
    hist_plot_c.pyplot(fig_c)

#TODO run the animation here
#TODO use this in parallel
#TODO refactor this to be a class

with st.spinner("Running MCMC..."):

    while st.session_state.running and st.session_state.idx < NUM_ITERATIONS:
        
        trace_fig_a, trace_ax_a = plt.subplots()
        sns.lineplot(x=x[:st.session_state.idx], y=y[:st.session_state.idx], ax=trace_ax_a)
        trace_plot_a.pyplot(trace_fig_a)

        trace_fig_b, trace_ax_b = plt.subplots()
        sns.lineplot(x=x[:st.session_state.idx], y=y[:st.session_state.idx], ax=trace_ax_b)
        trace_plot_b.pyplot(trace_fig_b)

        trace_fig_c, trace_ax_c = plt.subplots()
        sns.lineplot(x=x[:st.session_state.idx], y=y[:st.session_state.idx], ax=trace_ax_c)
        trace_plot_c.pyplot(trace_fig_c)

        fig_a, ax_a = plt.subplots()
        sns.histplot(y[:st.session_state.idx], ax=ax_a)
        hist_plot_a.pyplot(fig_a)

        fig_b, ax_b = plt.subplots()
        sns.histplot(y[:st.session_state.idx], ax=ax_b)
        hist_plot_b.pyplot(fig_b)

        fig_c, ax_c = plt.subplots()
        sns.histplot(y[:st.session_state.idx], ax=ax_c)
        hist_plot_c.pyplot(fig_c)
        
        st.session_state.idx += 1


