"""
The main plotting functions for the application.
The seaborn style was not available so it had to be applied manually.
"""

import altair as alt
import pandas as pd  # Ensure pandas is imported

import streamlit as st

class SeabornPlotStyles:
    """Collection of colours used by seaborn styles."""
    BACKGROUND_COLOR = "#EAEAF2" # seaborn style, dark grey
    AXIS_COLOR = "white" # white
    AXIS_LABEL_COLOR = "black"  # black color
    AXIS_TITLE_COLOR = "black"
    TITLE_COLOR = "#ECEFF4"


def trace_plot(data: pd.DataFrame) -> alt.Chart:
    """Create a trace plot of the parameters vs the number of iterations."""
    
    melted_data = data.reset_index().melt(
        id_vars=['index'],
        value_vars=['a', 'b', 'sigma'],
        var_name='parameter',
        value_name='value'
    )    
    
    trace_chart = alt.Chart(melted_data).mark_line(color='#e76f51').encode(
        x=alt.X('index', title='Number of Iterations'),
        y=alt.Y('value', title='Parameter Value'), 
        color='parameter:N'
    ).properties(title='Trace plot of parameters').configure_view(
        stroke='transparent',  # Remove the border around the chart
        fill=SeabornPlotStyles.BACKGROUND_COLOR
    ).configure_axis(
        grid=True,
        gridColor=SeabornPlotStyles.AXIS_COLOR, 
        domainColor=SeabornPlotStyles.AXIS_COLOR,
        tickColor=SeabornPlotStyles.AXIS_LABEL_COLOR,
        titleColor=SeabornPlotStyles.AXIS_TITLE_COLOR
    )#.configure_legend(
    #     titleColor=SeabornPlotStyles.AXIS_LABEL_COLOR,
    #     labelColor=SeabornPlotStyles.AXIS_LABEL_COLOR,
    #     title='Parameter'
    # )

    return trace_chart

def histogram_plot(data: pd.DataFrame, variable: str, title: str, bins: int = 10) -> alt.Chart:

    histogram = alt.Chart(data).mark_bar(opacity=0.8, color='#81A1C1', binSpacing=0).encode(
        x=alt.X(f'{variable}:Q', title=variable).bin(maxbins=bins),  # Changed to ordinal type for binned data
        y=alt.Y("count()", title="Frequency")
    ).properties(title=f"Distribution of {title}").configure_view(
        fill=SeabornPlotStyles.BACKGROUND_COLOR  # Background color
    ).configure_axis(
        grid=True,
        gridColor=SeabornPlotStyles.AXIS_COLOR,  # Grid lines color
        tickColor=SeabornPlotStyles.AXIS_LABEL_COLOR,
        labelColor=SeabornPlotStyles.AXIS_LABEL_COLOR
    )

    return histogram

def show_trace_plot() -> None:
    return st.altair_chart(
            altair_chart=trace_plot(st.session_state.trace_data), 
            use_container_width=True)

def show_histogram_plot(param: str) -> None:
    return st.altair_chart(
            altair_chart=histogram_plot(st.session_state.trace_data, param, param), 
            use_container_width=True)