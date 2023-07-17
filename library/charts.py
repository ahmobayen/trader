import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as ms


def candle_bar(data_chart: pd.DataFrame):
    """
    Creates a candlestick chart with volume bars.

    Args:
        data_chart (pd.DataFrame): DataFrame containing the financial data.

    Returns:
        go.Figure: Plotly Figure object representing the candlestick chart.

    """

    # Create a subplot with two rows
    fig = ms.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # Add candlestick trace to the first row
    fig.add_trace(
        go.Candlestick(
            x=data_chart.index,
            open=data_chart.Open,
            high=data_chart.High,
            low=data_chart.Low,
            close=data_chart.Close
        )
    )

    # Add volume bars trace to the second row
    fig.add_trace(
        go.Bar(x=data_chart.index, y=data_chart.Volume),
        row=2, col=1
    )

    # Update layout to hide the x-axis range slider
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


def line_chart(data_chart: pd.DataFrame, feature='Close'):
    """
    Creates a line chart for a specified feature in the financial data.

    Args:
        data_chart (pd.DataFrame): DataFrame containing the financial data.
        feature (str): Name of the feature column to be plotted (default: 'Close').

    Returns:
        go.Figure: Plotly Figure object representing the line chart.

    """

    # Create a new Figure object
    fig = go.Figure()

    # Add a scatter trace for the specified feature
    fig.add_trace(go.Scatter(x=data_chart.index, y=data_chart[feature], mode='lines', name='Stock Prices'))

    # Update the layout with title and axis labels
    fig.update_layout(
        title='Stock Prices History',
        xaxis=dict(title='Date'),
        yaxis=dict(title=feature)
    )
    return fig


def macd_chart(data_chart: pd.DataFrame):
    # Create a line plot for the 'macd' column
    line_trace = go.Scatter(x=data_chart.index, y=data_chart.MACD_12_26_9, name='MACD')
    Historgram = go.Bar(x=data_chart.index, y=data_chart.MACDh_12_26_9, name='Historgram')
    Signal = go.Scatter(x=data_chart.index, y=data_chart.MACDs_12_26_9, name='Signal')

    # Create the figure and add the traces
    fig = go.Figure(data=[line_trace, Historgram, Signal])

    # Add layout and axis labels
    fig.update_layout(title='Data with MACD', xaxis_title='Index', yaxis_title='Price')

    return fig


def model_evaluation(history):
    # Create the figure
    fig = go.Figure()

    # Add the training loss trace
    fig.add_trace(go.Scatter(
        x=list(range(len(history.history['loss']))),
        y=history.history['loss'],
        mode='lines',
        name='Training loss'
    ))

    # Add the validation loss trace
    fig.add_trace(go.Scatter(
        x=list(range(len(history.history['val_loss']))),
        y=history.history['val_loss'],
        mode='lines',
        name='Validation loss'
    ))

    # Set the layout of the plot
    fig.update_layout(
        title='Training and Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss'
    )

    # Display the plot
    fig.show()