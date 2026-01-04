from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import ScraperConfig


class XScraper:
    """Basic scraper and sentiment pipeline.

    Fetches JSON posts from a configurable endpoint, cleans the resulting text,
    applies VADER sentiment analysis, and generates summary reports.
    """

    def __init__(self, config: ScraperConfig):
        self.config = config
        self._analyzer = SentimentIntensityAnalyzer()
        self._session = requests.Session()

    def scrape(self) -> pd.DataFrame:
        """Download posts from the configured source and return a DataFrame."""

        response = self._session.get(self.config.source_url, timeout=15)
        response.raise_for_status()
        payload = response.json()

        normalized: List[Dict[str, str]] = []
        for raw_post in payload[: self.config.limit]:
            content = self._merge_fields(raw_post)
            normalized.append(
                {
                    "id": raw_post.get("id"),
                    "title": raw_post.get("title", ""),
                    "body": raw_post.get("body", ""),
                    "content": content,
                }
            )
        return pd.DataFrame(normalized)

    def clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply lightweight cleaning such as whitespace normalization."""

        clean_frame = frame.copy()
        clean_frame["clean_text"] = (
            clean_frame["content"].astype(str)
            .str.replace("\s+", " ", regex=True)
            .str.strip()
        )
        if self.config.min_chars:
            clean_frame = clean_frame[clean_frame["clean_text"].str.len() >= self.config.min_chars]
        clean_frame.reset_index(drop=True, inplace=True)
        return clean_frame

    def score_sentiment(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment scores and labels to the cleaned content."""

        scored = frame.copy()

        def _score_row(text: str) -> Tuple[float, str]:
            scores = self._analyzer.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            return compound, label

        scored[["sentiment_score", "sentiment"]] = scored["clean_text"].apply(
            lambda text: pd.Series(_score_row(text))
        )
        return scored

    def generate_reports(self, frame: pd.DataFrame) -> Dict[str, Path]:
        """Persist detailed data and aggregated sentiment counts."""

        output_dir = self.config.resolved_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        detail_path = output_dir / self.config.detail_filename
        frame.to_csv(detail_path, index=False)

        summary = frame["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
        summary_path = output_dir / self.config.report_filename
        summary.to_csv(summary_path, index=False)

        return {
            "detail_csv": detail_path,
            "sentiment_report_csv": summary_path,
        }

    @staticmethod
    def _merge_fields(raw_post: Dict[str, str]) -> str:
        title = raw_post.get("title", "")
        body = raw_post.get("body", "")
        return f"{title}\n{body}".strip()
