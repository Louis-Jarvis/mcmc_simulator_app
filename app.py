import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

import pathlib

from src.text import PROBLEM_DESCRIPTION, PROPOSAL_DISTRIBUTION_TEXT

NUM_ITERATIONS = 100

logging.basicConfig(level=logging.INFO) #TODO change
logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid")

#TODO eventually get rid of this.
x = np.linspace(0, 10, NUM_ITERATIONS)
y = np.sin(x) + np.random.normal(0, 0.1, NUM_ITERATIONS)

def update_trace_plot(x, y, idx, ax):
    ax.set_xlim(0, 10)
    ax.plot(x[:idx], y[:idx])
    #sns.lineplot(x=x[:idx], y=y[:idx], ax=ax)  # Set x-limits

def update_hist_plot(y, idx, ax):
    #ax.clear()
    ax.hist(y[:idx])

def update_all_plots(idx):
    update_trace_plot(x, y, idx, st.session_state.axs[0])
    for j in range(1, 4):
        update_hist_plot(y, idx, st.session_state.axs[j])

# initialisation
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.idx = 0

if "axs" not in st.session_state:
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Top row:
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('Trace Plot')
    ax1.set_xlim(0, NUM_ITERATIONS)

    # Bottom row:
    axs = [ax1]
    for j in range(3):
        ax = fig.add_subplot(gs[1, j])
        axs.append(ax)

    plt.suptitle('Metropolis Hastings Parameters', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title

    st.session_state.plots = fig
    st.session_state.axs = axs
    

#TODO add dependency injection
def start_button():
    st.session_state.running = True
    logger.info("Starting")

def stop_button():
    st.session_state.running = False

def reset_button():
    st.session_state.running = False
    st.session_state.idx = 0
    [ax.clear() for ax in st.session_state.axs]


st.title("Bayesian Linear Regression with MCMC")

#############################################
# Define the sidebar to control the animation
#############################################
st.sidebar.title("Inputs")

with st.sidebar:

    main_button = st.empty()
    reset_button_ = st.empty()

    if not st.session_state.running:
        logger.debug("Starting")
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

############################################
# Run the simulation and display the plots
############################################
with st.container():
    trace_plot_a = st.empty()

with st.spinner("Running MCMC..."):

    for i in range(0, NUM_ITERATIONS, 10):

        if not st.session_state.running:
            break

        update_trace_plot(x, y, i, st.session_state.axs[0])

        update_hist_plot(y, i, st.session_state.axs[1])
        update_hist_plot(y, i, st.session_state.axs[2])
        update_hist_plot(y, i, st.session_state.axs[3])
        
        plt.close()
        trace_plot_a.pyplot(st.session_state.plots)

        st.session_state.idx += 1

