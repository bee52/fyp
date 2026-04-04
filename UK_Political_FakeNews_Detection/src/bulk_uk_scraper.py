import argparse
import newspaper
import pandas as pd
from datetime import datetime
import time
import random
import logging

try:
    from .config import ensure_directory, load_config
    from .schema import normalize_article_record
except ImportError:
    from config import ensure_directory, load_config
    from schema import normalize_article_record

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rotating user agents for dynamic header switching
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
]

# Expanded UK news sources
TARGETS = [
    {'name': 'BBC_Politics', 'url': 'https://www.bbc.co.uk/news/politics'},
    {'name': 'Guardian_Politics', 'url': 'https://www.theguardian.com/politics'},
    {'name': 'Telegraph_Politics', 'url': 'https://www.telegraph.co.uk/politics'},
    {'name': 'Sky_Politics', 'url': 'https://news.sky.com/politics'},
    {'name': 'Independent_Politics', 'url': 'https://www.independent.co.uk/news/uk/politics'},
    {'name': 'Mail_Politics', 'url': 'https://www.dailymail.co.uk/news/politics/index.html'},
]

def get_random_user_agent():
    """Get a random user agent for dynamic header switching"""
    return random.choice(USER_AGENTS)

def rate_limited_sleep(base_delay=2.0):
    """Add random jitter to rate limiting to avoid predictable patterns"""
    jitter = random.uniform(0.5, 1.5)
    time.sleep(base_delay * jitter)

def scrape_with_retry(article, max_retries=3):
    """Retry failed downloads with exponential backoff"""
    for attempt in range(max_retries):
        try:
            article.download()
            article.parse()
            return True
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            if attempt < max_retries - 1:
                logger.info(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.warning(f"Failed to scrape after {max_retries} attempts: {e}")
                return False
    return False

def create_config():
    """Create newspaper config with dynamic user agent and optimized settings"""
    config = newspaper.Config()
    config.browser_user_agent = get_random_user_agent()
    config.request_timeout = 15
    config.number_threads = 2  # Parallel downloads (respectful)
    return config

def scrape_category(source, max_articles=200, base_delay=2.0, max_retries=3):
    print(f"----Scanning {source['name']}----")
    config = create_config()
    
    try:
        paper = newspaper.build(source['url'], config=config, memoize_articles=False)
        logger.info(f"Successfully built paper for {source['name']}: {len(paper.articles)} articles found")
    except Exception as e:
        logger.error(f"Error connecting to {source['name']}: {e}")
        return []

    article_data = []
    success_count = 0
    failed_count = 0

    for article in paper.articles:
        if success_count >= max_articles:
            break

        try:
            # Rate limited sleep with random jitter
            rate_limited_sleep(base_delay=base_delay)
            
            # Retry with exponential backoff
            if scrape_with_retry(article, max_retries=max_retries):
                # Filter to ensure article mentions UK politics keywords
                if any(x in article.text for x in ['Sunak', 'Starmer', 'MP', 'Parliament', 'Government']):
                    article_data.append(normalize_article_record({
                        'title': article.title,
                        'text': article.text,
                        'source': source['name'],
                        'date': datetime.now().strftime("%Y-%m-%d"),
                        'label': 0
                    }))
                    success_count += 1
                    print(f"[{success_count}] Collected: {article.title[:50]}...")
            else:
                failed_count += 1
                
        except Exception as e:
            logger.warning(f"Unexpected error processing article: {e}")
            failed_count += 1

    logger.info(f"{source['name']}: {success_count} collected, {failed_count} failed")
    return article_data

def parse_args():
    parser = argparse.ArgumentParser(description="Scrape UK political real-news sources")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--max-articles", type=int, default=None, help="Max articles per source")
    parser.add_argument("--base-delay", type=float, default=None, help="Rate limit base delay in seconds")
    parser.add_argument("--max-retries", type=int, default=None, help="Retries per article")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    scraper_cfg = cfg["scraping"]["real"]
    output_dir = args.output_dir or cfg["scraping"]["default_output_dir"]
    max_articles = args.max_articles if args.max_articles is not None else int(scraper_cfg["max_articles_per_source"])
    base_delay = args.base_delay if args.base_delay is not None else float(scraper_cfg["base_delay_seconds"])
    max_retries = args.max_retries if args.max_retries is not None else int(scraper_cfg["max_retries"])

    all_articles = []
    for source in TARGETS:
        data = scrape_category(source, max_articles=max_articles, base_delay=base_delay, max_retries=max_retries)
        all_articles.extend(data)

    if not all_articles:
        logger.error("No articles collected. Check your internet or try fewer targets.")
        return

    # Save to CSV using normalized contract and config-driven output directory.
    out_dir = ensure_directory(output_dir)
    df = pd.DataFrame(all_articles)
    filename = out_dir / f"uk_politics_{datetime.now().strftime('%Y%m%d')}.csv"
    df = df[["title", "text", "source", "date", "label"]]
    df.to_csv(filename, index=False)
    
    logger.info(f"SUCCESS: Saved {len(df)} UK articles to {filename}")
    print(f"\nSUCCESS: Saved {len(df)} UK articles to {filename}")
    print("Next Step: Merge this with your satire CSV to create the training set.")

if __name__ == "__main__":
    main()