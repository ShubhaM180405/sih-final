import re
import plotly
import os

def match_keywords(text: str, keywords: list) -> int:
    """
    Count exact keyword matches in text using word boundaries for precision.
    """
    text_lower = text.lower()
    count = 0
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        count += len(re.findall(pattern, text_lower))
    return count

def save_chart_as_image(fig, filename: str) -> str:
    """
    Save a Plotly figure as a PNG image and return the file path.
    """
    try:
        plotly.io.write_image(fig, file=filename, format="png")
        return filename
    except Exception as e:
        print(f"⚠️ Could not save chart {filename}: {e}")
        return ""
