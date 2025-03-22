"""
The main plotting functions for the application.
The seaborn style was not available so it had to be applied manually.
"""

import altair as alt
import pandas as pd


class SeabornPlotStyles:
    """Collection of colours used by seaborn styles."""
    BACKGROUND_COLOR = "#EAEAF2"  # dark grey
    AXIS_COLOR = "white"  # white
    AXIS_LABEL_COLOR = "black"
    AXIS_TITLE_COLOR = "black"
    TITLE_COLOR = "#ECEFF4"  # light grey

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
    )

    return trace_chart

def histogram_plot(
        data: pd.DataFrame, 
        variable: str, 
        title: str, 
        bins: int = 10
        ) -> alt.Chart:
    """Histogram of theta parameter."""
    histogram = alt.Chart(data).mark_bar(
        opacity=0.8, 
        color='#81A1C1', 
        binSpacing=0
        ).encode(
            x=alt.X(f'{variable}:Q', title=variable).bin(maxbins=bins), 
            y=alt.Y("count()", title="Frequency")
    ).properties(title=f"Distribution of {title}").configure_view(
        fill=SeabornPlotStyles.BACKGROUND_COLOR  # Background color
    ).configure_axis(
        grid=True,
        gridColor=SeabornPlotStyles.AXIS_COLOR,
        tickColor=SeabornPlotStyles.AXIS_LABEL_COLOR,
        labelColor=SeabornPlotStyles.AXIS_LABEL_COLOR
    )

    return histogram

def plot_histogram_or_empty(
        current_data: pd.DataFrame, 
        variable: str, 
        title: str) -> None:
    """Optionally plot a histogram of the thetas if there is data available. Ignoring 
    the initial theta estimates."""
    if len(current_data) > 1: # do not plot the initial theta estimates        
        return histogram_plot(current_data, variable, title)
    
    return alt.Chart(pd.DataFrame()).mark_bar().encode()