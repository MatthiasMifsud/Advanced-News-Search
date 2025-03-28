from combiner import NewsSearchApp

if __name__ == "__main__":
    api_key = 'ba980974734d4bec81edfdf6412877c5'
    source_list = [
        "bbc", "cnn", "nytimes", "theguardian", "reuters", "foxnews",
        "bbc-news", "cbs-news", "abc-news", "huffpost"
        ]

    combiner = NewsSearchApp(api_key, source_list)
    combiner.runner()