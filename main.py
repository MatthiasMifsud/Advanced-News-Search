from combiner import NewsSearchApp

if __name__ == "__main__":
    api_key = '4877741a5fa94bb3b578f97b53671e71'
    source_list = [
        "bbc", "cnn", "nytimes", "theguardian", "reuters", "foxnews",
        "bbc-news", "cbs-news", "abc-news", "huffpost"
        ]

    combiner = NewsSearchApp(api_key, source_list)
    combiner.runner()