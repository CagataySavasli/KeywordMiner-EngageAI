from googlenewsdecoder import new_decoderv1
from gnews import GNews
import pandas as pd
import newspaper
import ssl
import urllib.request

# SSL doğrulamasını atlamak için:
ssl._create_default_https_context = ssl._create_unverified_context

key_words = [
    'US elections',
    'Donald Trump',
    'Kemala Harris'
]

errors = []
google_news = GNews()
google_news.max_results = 1000
google_news.start_date = (2024, 10, 10)
google_news.end_date = (2024, 11, 24)
df = pd.DataFrame(columns=['date', 'key_word', 'title', 'text', 'description', 'publisher'])

for key_word in key_words:
    news = google_news.get_news(key_word)
    for new in news:
        try:
            interval_time = 5 # this can change to any time, but 5 is recommended
            decoded_url = new_decoderv1(new['url'], interval=interval_time)
            new['url'] = decoded_url['decoded_url']
            artic = google_news.get_full_article(new['url'])
            if len(artic.text) == 0:
                continue
            row = pd.DataFrame.from_dict({'date': [new['published date']],
                                          'key_word' : [key_word],
                                          'title': [new['title']],
                                          'text': [artic.text],
                                          'description': [new['description']],
                                          'publisher': [new['publisher']['title']],
                                          'url':[new['url']]})

            df = pd.concat([df, row], ignore_index=True)
            print("\r",( key_words.index(key_word)+1), "/", len(key_words), " - ", (news.index(new)+1), "/", len(news), end=" ")
        except:
            errors.append(new)
    print(f"----- {key_word} -----")

df.to_csv('src/datas/us_elections_news.csv', index=False)