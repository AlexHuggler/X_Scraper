# X/Twitter Scraper

A production-ready, Playwright-based tool for scraping posts from X.com (Twitter) accounts.

## Features

- **Playwright-powered**: Uses Playwright for robust browser automation
- **Headful by default**: Watch the scraping happen, with headless mode available
- **Resilient scraping**: Handles infinite scroll, lazy loading, and DOM changes
- **Rate limit handling**: Automatic backoff and retry on rate limits
- **Session support**: Save and reuse login sessions for authenticated scraping
- **Flexible output**: JSON and CSV formats, per-account and combined files
- **Configurable**: Easy JSON config file with CLI overrides
- **Comprehensive logging**: Console + optional file logging

## Project Structure

```
X_Scraper/
├── scrape.py              # Main entry point (run this)
├── config.json            # Sample configuration file
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── x_scraper/             # Main package
│   ├── __init__.py
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration handling
│   ├── scraper.py         # Core scraping logic
│   ├── extractors.py      # Data extraction with fallbacks
│   ├── session.py         # Session/auth management
│   ├── output.py          # JSON/CSV output handlers
│   └── logger.py          # Logging configuration
├── data/                  # Default output directory
└── logs/                  # Log files (when enabled)
```

## Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **pip** (Python package manager)

## Docker Quick Start

For the fastest setup with all dependencies pre-installed, use Docker. This method provides an isolated environment with Playwright and all required tools ready to use.

### Prerequisites

- **Docker** (with Docker Compose)
- **Git** (optional, for cloning)

### Quick Setup

1. **Clone or download the repository**

   ```bash
   git clone <repository-url>
   cd X_Scraper
   ```

2. **Build and start the container**

   ```bash
   docker-compose up --build
   ```

   This builds the image and starts an interactive container with bash shell.

3. **Access the scraper shell**

   ```bash
   docker exec -it scraper bash
   ```

   You're now in the container with all dependencies ready.

### Using the Scraper

Inside the container shell, convenient aliases are available:

```bash
# Scrape accounts with opts
scrape --accounts elonmusk,OpenAI --limit 10

# Scrape accounts with config file
scrape --config config.json

# Get help
scrape --help
```

### Authentication

Scraping works without login, but for authenticated access (recommended to avoid rate limits):

**If you have Playwright installed on your host machine:**

```bash
# On host (outside Docker):
python scrape.py --login
```

Complete login in browser, then the session is available in Docker.

**Manual session setup (no Playwright required):**

1. Install a browser extension like "Cookie-Editor" or "Storage Manager"
2. Log in to x.com in your browser
3. Use the extension to export cookies for domains `x.com'
4. Create `.x_session/storage_state.json` with:

```json
{
  "cookies": [
    {
      "name": "auth_token",
      "value": "your_auth_token_value",
      "domain": ".x.com",
      "path": "/",
      "httpOnly": true,
      "secure": true
    }
    // Include all x.com cookies
  ],
  "origins": []
}
```

### Data Persistence

Scraped data, logs, and sessions persist on your host:

- `data/` - Scraped posts (JSON/CSV)
- `logs/` - Log files  
- `.x_session/` - Login sessions

## Manual Installation

### 1. Clone or download the repository

```bash
cd X_Scraper
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Playwright browsers

```bash
playwright install chromium
```

## Quick Start

### Basic usage with config file

```bash
# Edit config.json with your accounts first
python scrape.py --config config.json
```

### Direct command-line usage

```bash
python scrape.py --accounts elonmusk,OpenAI --limit 25 --out ./data
```

### Run in headless mode

```bash
python scrape.py --config config.json --headless
```

## Configuration

### Config File (config.json)

Edit `config.json` to customize your scraping:

```json
{
  "accounts": [
    "elonmusk",
    "OpenAI",
    "anthropikiw"
  ],
  "posts_per_account": 25,
  "output_dir": "./data",
  "date_cutoff_days": 30,
  "headless": false,
  "slow_mo": 50,
  "max_retries": 3,
  "retry_delay": 2.0,
  "scroll_delay_min": 1.5,
  "scroll_delay_max": 3.0,
  "page_timeout": 30000,
  "element_timeout": 10000,
  "session_file": null,
  "log_file": "./logs/scraper.log",
  "log_level": "INFO"
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `accounts` | array | `[]` | List of X usernames or profile URLs to scrape |
| `posts_per_account` | int | `20` | Maximum posts to collect per account |
| `output_dir` | string | `./data` | Directory to save output files |
| `date_cutoff_days` | int | `null` | Only scrape posts from last N days (null = no limit) |
| `headless` | bool | `false` | Run browser without visible window |
| `slow_mo` | int | `50` | Milliseconds between browser actions |
| `max_retries` | int | `3` | Max retry attempts on failure |
| `retry_delay` | float | `2.0` | Base delay (seconds) for exponential backoff |
| `scroll_delay_min` | float | `1.5` | Minimum delay between scrolls |
| `scroll_delay_max` | float | `3.0` | Maximum delay between scrolls |
| `page_timeout` | int | `30000` | Page load timeout (ms) |
| `element_timeout` | int | `10000` | Element wait timeout (ms) |
| `session_file` | string | `null` | Path to saved session file |
| `log_file` | string | `null` | Path to log file (optional) |
| `log_level` | string | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Account Format

Accounts can be specified in multiple formats:

```json
{
  "accounts": [
    "username",
    "@username",
    "https://x.com/username",
    "https://twitter.com/username"
  ]
}
```

## CLI Reference

```bash
python scrape.py [OPTIONS]
```

### Options

| Flag | Description |
|------|-------------|
| `--config, -c FILE` | Path to configuration JSON file |
| `--accounts, -a LIST` | Comma-separated list of accounts |
| `--out, -o DIR` | Output directory |
| `--limit, -n NUM` | Max posts per account (overrides config) |
| `--headless` | Run in headless mode |
| `--headful` | Run in headful mode (visible browser) |
| `--days NUM` | Only scrape posts from last N days |
| `--session FILE` | Path to session storage file |
| `--log-file FILE` | Path to log file |
| `--verbose, -v` | Enable debug logging |
| `--quiet, -q` | Suppress non-essential output |
| `--login` | Interactive login mode |
| `--verify-session` | Verify saved session |
| `--init-config` | Create sample config.json |
| `--help, -h` | Show help message |

### Examples

```bash
# Basic scraping with config
python scrape.py --config config.json

# Override config settings
python scrape.py --config config.json --limit 50 --headless

# Quick scrape without config file
python scrape.py --accounts elonmusk,OpenAI,anthropikiw --limit 20

# Scrape with date cutoff (last 7 days)
python scrape.py --config config.json --days 7

# Enable verbose logging
python scrape.py --config config.json --verbose

# Create a new config file
python scrape.py --init-config
```

## Authentication (Optional)

By default, the scraper works as a logged-out user. For authenticated scraping:

### 1. Save your session

```bash
python scrape.py --login
```

This opens a browser window. Log in manually, then press Enter in the terminal. Your session is saved to `.x_session/storage_state.json`.

### 2. Use the saved session

Future runs automatically detect and use the saved session. Or specify explicitly:

```bash
python scrape.py --config config.json --session ./.x_session/storage_state.json
```

### 3. Verify session validity

```bash
python scrape.py --verify-session
```

### Security Notes

- **Never hardcode credentials** - The scraper never stores passwords
- Sessions are stored as browser state files
- Add `.x_session/` to your `.gitignore`
- Sessions may expire after some time

## Output

### File Structure

After scraping, the output directory contains:

```
data/
├── results.json           # Combined results (all accounts)
├── results.csv            # Combined results CSV
├── elonmusk.json          # Per-account JSON
├── elonmusk.csv           # Per-account CSV
├── OpenAI.json
├── OpenAI.csv
└── ...
```

### Data Fields

Each post contains:

| Field | Type | Description |
|-------|------|-------------|
| `account_handle` | string | Twitter handle (e.g., "elonmusk") |
| `account_display_name` | string | Display name |
| `post_url` | string | Full URL to the post |
| `post_id` | string | Unique post ID |
| `timestamp` | string | ISO 8601 timestamp |
| `text_content` | string | Post text |
| `reply_count` | int | Number of replies |
| `repost_count` | int | Number of reposts |
| `like_count` | int | Number of likes |
| `view_count` | int | Number of views |
| `is_repost` | bool | Whether this is a repost |
| `is_quote` | bool | Whether this quotes another post |
| `media_urls` | array | URLs of attached media |
| `scraped_at` | string | When the data was scraped |

### Sample JSON Output

```json
[
  {
    "account_handle": "elonmusk",
    "account_display_name": "Elon Musk",
    "post_url": "https://x.com/elonmusk/status/1234567890",
    "post_id": "1234567890",
    "timestamp": "2024-01-15T10:30:00.000Z",
    "text_content": "This is the tweet content...",
    "reply_count": 5000,
    "repost_count": 10000,
    "like_count": 50000,
    "view_count": 5000000,
    "is_repost": false,
    "is_quote": false,
    "media_urls": ["https://pbs.twimg.com/media/..."],
    "scraped_at": "2024-01-16T14:22:33.123456"
  }
]
```

## Troubleshooting

### Common Issues

#### "No accounts specified"

Make sure your config.json has accounts listed, or use `--accounts`:

```bash
python scrape.py --accounts username1,username2
```

#### "Account does not exist"

Check if the username is correct. The scraper handles:

- Suspended accounts
- Non-existent accounts
- Protected/private accounts (limited access)

#### Rate Limiting

If you see rate limit errors:

- Reduce `posts_per_account`
- Increase `scroll_delay_min` and `scroll_delay_max`
- Use `--headless` mode for faster recovery
- Consider using authenticated session

#### CAPTCHA Prompts

X may show CAPTCHAs to automated browsers:

- Use headful mode (default) to solve manually
- Try logging in first with `--login`
- Reduce scraping frequency

#### Selector Breakage

X frequently changes its DOM structure. If scraping fails:

1. Check if X.com loads normally in a regular browser
2. Update the selectors in `x_scraper/extractors.py`
3. Check for project updates

#### Browser Not Found

If Playwright can't find Chromium:

```bash
playwright install chromium
```

#### Permission Denied

On Linux/Mac, make the script executable:

```bash
chmod +x scrape.py
```

### Debug Mode

For detailed debugging information:

```bash
python scrape.py --config config.json --verbose
```

Or set in config.json:

```json
{
  "log_level": "DEBUG"
}
```

### Log Files

Enable log files for persistent debugging:

```bash
python scrape.py --config config.json --log-file ./logs/debug.log
```

## Legal Notice

This tool is for educational and research purposes. When using:

- Respect X's Terms of Service
- Don't scrape at high volumes
- Don't use for spam or harassment
- Be mindful of rate limits
- Only scrape public data unless authenticated

## License

MIT License - See LICENSE file for details.
