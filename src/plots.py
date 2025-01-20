"""
The main plotting functions for the application.
The seaborn style was not available so it had to be applied manually.
"""

import altair as alt
import pandas as pd  # Ensure pandas is imported

class SeabornPlotStyles:
    """Collection of colours used by seaborn styles."""
    BACKGROUND_COLOR = "#EAEAF2" # seaborn style, dark grey
    AXIS_COLOR = "white" # white
    AXIS_LABEL_COLOR = "black"  # black color
    AXIS_TITLE_COLOR = "black"
    TITLE_COLOR = "#ECEFF4"


def trace_plot(data):
    """Create a trace plot of the parameters vs the number of iterations."""
    trace_chart = alt.Chart(data).mark_line(color='#e76f51').encode(
        x=alt.X('x', title='Number of Iterations'),
        y=alt.Y('y', title='Parameter Value')
    ).properties(title='Trace plot of parameters').configure_view(
        stroke='transparent',  # Remove the border around the chart
        fill=SeabornPlotStyles.BACKGROUND_COLOR  # Dark Slate Gray background
    ).configure_axis(
        grid=True,
        gridColor=SeabornPlotStyles.AXIS_COLOR, 
        domainColor=SeabornPlotStyles.AXIS_COLOR,
        tickColor=SeabornPlotStyles.AXIS_LABEL_COLOR,
        titleColor=SeabornPlotStyles.AXIS_TITLE_COLOR
    )

    return trace_chart

def histogram_plot(data, variable, title, bins=10):
    # Bin the data using pandas.cut
    if data.empty:
        # cannot cut pd.cut to empty data
        histogram_data = pd.DataFrame(columns=['Bins', 'Counts'])
    else:
        histogram_data = pd.cut(data[variable], bins=bins, include_lowest=True).value_counts().sort_index().reset_index()
        histogram_data.columns = [variable, 'Counts']  # Rename columns to match expected format
        histogram_data['bin_centres'] = histogram_data[variable].apply(lambda x: (x.left + x.right) / 2)


    histogram = alt.Chart(histogram_data).mark_bar(opacity=0.8, color='#81A1C1', binSpacing=0).encode(
        x=alt.X('bin_centres:Q', title=variable),  # Changed to ordinal type for binned data
        y=alt.Y('Counts:Q', title=None)
    ).properties(title=f"Distribution of {variable}").configure_view(
        #stroke='transparent',  # Remove the border around the chart
        fill=SeabornPlotStyles.BACKGROUND_COLOR  # Background color
    ).configure_axis(
        grid=True,  # Show grid lines
        gridColor=SeabornPlotStyles.AXIS_COLOR,  # Grid lines color
        tickColor=SeabornPlotStyles.AXIS_LABEL_COLOR,
        labelColor=SeabornPlotStyles.AXIS_LABEL_COLOR
    )

    return histogram
