import pandas as pd
import io
from datetime import date as _date

def load_comments(uploaded_file) -> pd.DataFrame:
    """
    Robustly read uploaded file (Streamlit UploadedFile or a file path).
    Accepts: .csv, .txt, .xlsx, .xls, or raw bytes from Streamlit.
    Normalizes columns to: 'text', 'author', 'date' and returns a DataFrame
    with only those columns.
    """
    # --- get bytes and filename (support Streamlit UploadedFile) ---
    if hasattr(uploaded_file, "getvalue"):
        content_bytes = uploaded_file.getvalue()
        filename = getattr(uploaded_file, "name", "") or ""
    else:
        # treat uploaded_file as a path-like
        with open(uploaded_file, "rb") as f:
            content_bytes = f.read()
        filename = str(uploaded_file)

    name_lower = filename.lower()

    # --- Try to read as Excel ---
    if name_lower.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(io.BytesIO(content_bytes))
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {e}")

    # --- TXT: treat as one comment per line ---
    elif name_lower.endswith(".txt"):
        try:
            text = content_bytes.decode("utf-8", errors="replace")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            df = pd.DataFrame(lines, columns=["text"])
        except Exception as e:
            raise ValueError(f"Could not read text file: {e}")

    # --- CSV or unknown: try CSV parsing with fallbacks ---
    else:
        # 1) try pandas read_csv on bytes (fast path)
        try:
            df = pd.read_csv(io.BytesIO(content_bytes))
        except Exception:
            # 2) try reading with utf-8 string IO
            try:
                s = content_bytes.decode("utf-8", errors="replace")
                df = pd.read_csv(io.StringIO(s))
            except Exception:
                # 3) fallback: treat as plain text (one comment per line)
                try:
                    s = content_bytes.decode("utf-8", errors="replace")
                    lines = [l.strip() for l in s.splitlines() if l.strip()]
                    df = pd.DataFrame(lines, columns=["text"])
                except Exception as e:
                    raise ValueError(f"Could not parse uploaded file as CSV or text: {e}")

    # --- Normalize column names to lowercase stripped strings ---
    df.columns = df.columns.astype(str).str.strip().str.lower()

    # --- Detect text/comment column ---
    text_aliases = [
        "text", "comment", "comments", "feedback", "review", "message", "body", "content"
    ]
    text_col = next((c for c in df.columns if c in text_aliases), None)

    # If no known text column but only one column exists, assume it's the text column
    if text_col is None and df.shape[1] == 1:
        text_col = df.columns[0]

    if text_col is None:
        raise ValueError(
            "Uploaded file must contain a text column (e.g. 'text', 'comment', 'feedback', 'review')."
        )

    # --- Detect/normalize author column ---
    author_aliases = ["author", "user", "username", "name", "sender"]
    author_col = next((c for c in df.columns if c in author_aliases), None)

    # --- Detect/normalize date column ---
    date_aliases = ["date", "created_at", "timestamp", "time", "created", "submitted"]
    date_col = next((c for c in df.columns if c in date_aliases), None)

    # --- Rename mapped columns to standard names ---
    rename_map = {text_col: "text"}
    if author_col:
        rename_map[author_col] = "author"
    if date_col:
        rename_map[date_col] = "date"

    df = df.rename(columns=rename_map)

    # --- Ensure 'text' exists and is string ---
    df["text"] = df["text"].astype(str).fillna("").str.strip()

    # --- Fill or create 'author' ---
    if "author" not in df.columns:
        df["author"] = "Anonymous"
    else:
        df["author"] = df["author"].astype(str).fillna("").replace("", "Anonymous")

    # --- Fill or create 'date' and coerce to date ---
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().all():
            today = _date.today()
            df["date"] = pd.to_datetime(today)
        df["date"] = df["date"].dt.date
    else:
        df["date"] = _date.today()

    # --- Keep only the three columns and reset index ---
    df_out = df[["text", "author", "date"]].reset_index(drop=True)

    return df_out

def save_results_to_csv(df: pd.DataFrame, filename: str = "sentiment_results.csv") -> str:
    """Save results to CSV and return filename (local path)."""
    df.to_csv(filename, index=False)
    return filename
