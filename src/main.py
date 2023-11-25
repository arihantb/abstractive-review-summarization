import pandas as pd
import pickle
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from models import db, Product
from bson import ObjectId
from encoder_decoder import Model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from difflib import SequenceMatcher
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

df = pd.read_csv('Reviews.csv')
analyser = SentimentIntensityAnalyzer()


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return dict(list(score.items())[:-1])


def get_reviews(product_name):
    global df
    return list(df[df['1'] == product_name]['0'].astype(str))


@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    product_list = []
    for name in db.products.distinct("name"):
        product_list.append(name)

    return templates.TemplateResponse("index.html", {"request": request, "product_list": product_list})


@app.get("/product", response_class=HTMLResponse)
def read_product(request: Request, q: str):
    products = []

    for product in db.products.find({"name": {"$regex": q, "$options": "i"}}):
        products.append(Product(**product))
    return templates.TemplateResponse("result.html", {"request": request, "products": products, "q": q})


@app.get("/product/review/{id}", response_class=HTMLResponse)
def read_review(request: Request, id: str):
    response = db.products.find_one({"_id": ObjectId(id)})

    print('===========================================')
    print(response)

    # reviews_list = get_reviews(response["name"])

    # reviews_list = []
    # products = ['B007JFMH8M', 'B002QWP89S', 'B0026RQTGE', 'B002QWHJOU', 'B002QWP8H0', 'B003B3OOPA', 'B001EO5Q64', 'B0013NUGDE', 'B007M83302', 'B000VK8AVK']

    # with open('Reviews.csv', 'r', encoding='utf-8') as f:
    #     for line in f.readlines()[1:]:
    #         if line.split(',')[1] == response['product_id']:
    #             reviews_list.append(line.split(',')[-1])

    df = pd.read_csv('Reviews.csv')

    print(df['ProductId'])
    print('===========================================')
    print(df[df['ProductId'] == response['product_id']])

    reviews_list = df[df['ProductId'] == response['product_id']]['Text'].to_list()

    with open('x_tokenizer', 'rb') as f:
        tokenizer = pickle.load(f)

    input_seq = tokenizer.texts_to_sequences(reviews_list)
    input_seq = pad_sequences(input_seq, maxlen=80, padding='post')

    decoded_sentences, pos_abstract_summary, neg_abstract_summary = [], [], []
    pos_sentiment, neg_sentiment = 0, 0

    def match(a, b):
        return SequenceMatcher(None, a, b).ratio()

    model = Model()

    for i in tqdm(range(50)):
        try:
            summary = model.decode_sequence(input_seq[i].reshape(1, 80)).strip()
            decoded_sentences.append(
                {summary: sentiment_analyzer_scores(summary)})
        except IndexError:
            pass

    for i in range(len(decoded_sentences)):
        for j in range(len(decoded_sentences)):
            if not decoded_sentences[i] == decoded_sentences[j]:
                pos_sentiment += [*decoded_sentences[i].values()][0]['pos']
                neg_sentiment += [*decoded_sentences[i].values()][0]['neg']

                # if [*decoded_sentences[i].keys()][0] == 'five stars':
                #     pos_abstract_summary.append([*decoded_sentences[i].keys()][0])
                #     break
                # elif [*decoded_sentences[i].keys()][0] == 'four stars':
                #     pos_abstract_summary.append([*decoded_sentences[i].keys()][0])
                #     break
                # elif [*decoded_sentences[i].keys()][0] == 'three stars':
                #     pos_abstract_summary.append([*decoded_sentences[i].keys()][0])
                #     break
                # elif [*decoded_sentences[i].keys()][0] == 'two stars':
                #     neg_abstract_summary.append([*decoded_sentences[i].keys()][0])
                #     break
                # elif [*decoded_sentences[i].keys()][0] == 'one star':
                #     neg_abstract_summary.append([*decoded_sentences[i].keys()][0])
                #     break

                if match([*decoded_sentences[i].keys()][0], [*decoded_sentences[j].keys()][0]) < 0.5:
                    if [*decoded_sentences[i].values()][0]['neg'] < [*decoded_sentences[i].values()][0]['pos']:
                        pos_abstract_summary.append([*decoded_sentences[i].keys()][0])
                    else:
                        neg_abstract_summary.append([*decoded_sentences[i].keys()][0])

                    break

    product = Product(**response)
    pos_sentiment = pos_sentiment / len(decoded_sentences)
    neg_sentiment = neg_sentiment / len(decoded_sentences)
    positive_sentiment = pos_sentiment + (1 - (pos_sentiment + neg_sentiment)) / 2
    negative_sentiment = neg_sentiment + (1 - (pos_sentiment + neg_sentiment)) / 2

    return templates.TemplateResponse("review.html", {"request": request, "product": product,
                                                      "positive_summary": '. '.join([i.capitalize() for i in list
                                                          (set(pos_abstract_summary))]) + '.',
                                                      "negative_summary": '. '.join([i.capitalize() for i in list
                                                          (set(neg_abstract_summary))]) + '.',
                                                      "positive_sentiment": positive_sentiment * 100,
                                                      "negative_sentiment": negative_sentiment * 100})
