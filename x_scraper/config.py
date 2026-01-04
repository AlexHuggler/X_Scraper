from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScraperConfig:
    """Configuration for fetching, cleaning, and reporting scraped content."""

    source_url: str = "https://jsonplaceholder.typicode.com/posts"
    limit: int = 20
    min_chars: int = 40
    output_dir: Path = Path("artifacts")
    detail_filename: str = "scraped_posts.csv"
    report_filename: str = "sentiment_report.csv"

    def resolved_output_dir(self) -> Path:
        """Return a fully expanded output directory path."""

        return Path(self.output_dir).expanduser().resolve()
