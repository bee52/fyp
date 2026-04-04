import newspaper
import nltk

nltk.download('punkt')

def test_uk_scraper():

    url = "https://www.bbc.com/news/politics"

    article = newspaper.Article(url)

    try:
        print(f"Attempting to scrape: {url}")
        article.download()
        article.parse()
        article.nlp()

        print("\n--- SCRAPING SUCCESS ---")
        print(f"Title: {article.title}")
        print(f"Authors: {article.authors}")
        print(f"Publish Date: {article.publish_date}")
        print(f"Summary: {article.summary}")

        #saving text file for 'corpus'
        with open("data/external/scraped_sample.txt", "w", encoding='utf-8') as f:
            f.write(f"{article.title}\n\n{article.text}")
            print("\nSaved article to data/external/scraped_sample.txt")

    except Exception as e:
        print(f"Error scraping: {e}")
    
if __name__ == "__main__":
    test_uk_scraper()
