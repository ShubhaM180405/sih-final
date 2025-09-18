import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def sentiment_distribution(df: pd.DataFrame):
    """Donut chart of sub-sentiment distribution"""
    if df.empty:
        return go.Figure()
    fig = px.pie(
        df, names="sentiment_sub", hole=0.5,
        color="sentiment_sub",
        color_discrete_map={
            "Positive": "lightblue",
            "Negative": "red",
            "Neutral (Pure Neutral)": "grey",
            "Neutral (Dominantly Negative)": "orange",
            "Neutral (Dominantly Positive)": "green"
        }
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(title="😊 Sub-Sentiment Distribution")
    return fig

def sentiment_over_time(df: pd.DataFrame):
    """Scatter plot of sub-sentiments over time"""
    if df.empty or "date" not in df.columns:
        return go.Figure()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    fig = px.scatter(
        df, x="date", y="sentiment_sub", color="sentiment_sub",
        hover_data=["text"],
        color_discrete_map={
            "Positive": "lightblue",
            "Negative": "red",
            "Neutral (Pure Neutral)": "grey",
            "Neutral (Dominantly Negative)": "orange",
            "Neutral (Dominantly Positive)": "green"
        }
    )
    fig.update_layout(title="📈 Sub-Sentiment Over Time", yaxis_title="Sentiment")
    return fig

def sentiment_main_sub_breakdown(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart showing main sentiment counts with sub-sentiment breakdown."""
    if df.empty:
        return go.Figure()
    grouped = df.groupby(['sentiment_main', 'sentiment_sub']).size().unstack(fill_value=0)
    fig = go.Figure()
    for sub_sentiment in grouped.columns:
        fig.add_trace(go.Bar(
            name=sub_sentiment,
            x=grouped.index,
            y=grouped[sub_sentiment],
            text=grouped[sub_sentiment],
            textposition='auto',
            marker=dict(
                color={
                    "Positive": "lightblue",
                    "Negative": "red",
                    "Neutral (Pure Neutral)": "grey",
                    "Neutral (Dominantly Negative)": "orange",
                    "Neutral (Dominantly Positive)": "green"
                }.get(sub_sentiment, "grey")
            )
        ))
    fig.update_layout(
        title="📊 Main Sentiment with Sub-Sentiment Breakdown",
        xaxis_title="Main Sentiment",
        yaxis_title="Count",
        barmode='stack',
        template="plotly_dark",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white")
    )
    return fig
