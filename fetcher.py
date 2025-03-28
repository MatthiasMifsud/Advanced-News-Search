from tqdm import tqdm
import requests
import concurrent.futures

class News_API: 
    def __init__(self, api_key, source_list):
        self.api_key = api_key
        self.source_list = source_list
        self.articles = []

    def fetch_articles(self, source):
        try:
            url = f'https://newsapi.org/v2/everything?q={source}&apiKey={self.api_key}'
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200 and data.get('status') == 'ok':
                 articles = data['articles']
                 self.articles.extend(articles)
            else:
                print(f"Failed to fetch articles from {source}. API response: {data}")
        
        except Exception as e:
            print(f"Error fetching articles from {source} due to ERROR: {e}")
             
    def parallel_news_fetching(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(self.fetch_articles, self.source_list), total=len(self.source_list), desc="Comparing News Articles"))
