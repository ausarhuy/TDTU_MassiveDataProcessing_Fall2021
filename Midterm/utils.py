import json
from datetime import datetime
from urllib.request import urlopen

import numpy as np
import requests
import spacy
from bs4 import BeautifulSoup
from dateutil import parser
from numpy import zeros
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode

nlp = spacy.load("en_core_web_lg")


def preprocessing(sentence, mode='text'):
	# remove punctuation and lower case tokens
	doc = nlp(sentence.translate(str.maketrans('', '', '.,')).lower())
	# convert token to pos token
	pos_sentence = list(map(lambda token: token.pos_, doc))
	if mode == 'token':
		return pos_sentence
	elif mode == 'text':
		return ' '.join(pos_sentence)
	else:
		raise 'wrong mode format'


def find_similar_author(df, main, comparator, model):
	if 'Tfidf' in str(model):
		main_embed_quotes = df.quote[df.name == main].apply(lambda x: model.transform([preprocessing(x)]).toarray()).values
		main_vec = np.sum(main_embed_quotes, axis=0)/len(main_embed_quotes)
		comparator_embed_quotes = df.quote[df.name == comparator].apply(lambda x: model.transform([preprocessing(x)]).toarray()).values
		comparator_vec = np.sum(comparator_embed_quotes, axis=0)/len(comparator_embed_quotes)
		return np.squeeze(cosine_similarity(main_vec, comparator_vec))
	else:
		main_embed_quotes = df.quote[df.name == main].apply(
			lambda x: get_feature_vector(x, model))
		main_vec = np.sum(main_embed_quotes, axis=0) / len(main_embed_quotes)
		comparator_embed_quotes = df.quote[df.name == comparator].apply(
			lambda x: get_feature_vector(x, model))
		comparator_vec = np.sum(comparator_embed_quotes, axis=0) / len(comparator_embed_quotes)
		return np.squeeze(cosine_similarity(main_vec, comparator_vec))


def get_similar_by_quote(quote, other_quote, model, name):
	if name == 'word2vec':
		quote_vec = get_feature_vector(quote, model)
		other_quote_vec = get_feature_vector(other_quote, model)
		return np.squeeze(cosine_similarity([quote_vec], [other_quote_vec]))
	elif name == 'tfidf':
		quote_vec = model.transform([preprocessing(quote)]).toarray()
		other_quote_vec = model.transform([preprocessing(other_quote)]).toarray()
		return np.squeeze(cosine_similarity(quote_vec, other_quote_vec))
	else:
		raise 'wrong model'


def get_feature_vector(sentence, word2vec):
	preprocessed_tokens = preprocessing(sentence, mode='token')
	vector = zeros(word2vec.vector_size).T
	count = 0
	for token in preprocessed_tokens:
		if token in word2vec.wv:
			vector += word2vec.wv[token]
			count += 1
	if count == 0:
		return vector
	return vector / count


def id_for_page(page):
	"""Uses the wikipedia api to find the wikidata id for a page"""
	api = "https://en.wikipedia.org/w/api.php"
	query = "?action=query&prop=pageprops&titles=%s&format=json"
	slug = page.split('/')[-1]
	response = json.loads(requests.get(api + query % slug).content)
	# Assume we got 1 page result and it is correct.
	page_info = list(response['query']['pages'].values())[0]
	return page_info['pageprops']['wikibase_item']


def lifespan_for_id(wikidata_id, claim_id='P570'):
	"""Uses the wikidata API to retrieve wikidata for the given id."""
	data_url = "https://www.wikidata.org/wiki/Special:EntityData/%s.json"
	page = json.loads(requests.get(data_url % wikidata_id).content)

	claims = list(page['entities'].values())[0]['claims']
	# P569 (birth) and P570 (death) ... not everyone has died yet.
	return get_claim_as_time(claims, claim_id)


def get_claim_as_time(claims, claim_id):
	"""Helper function to work with data returned from wikidata api"""
	try:
		claim = claims[claim_id][0]['mainsnak']['datavalue']
		assert claim['type'] == 'time', "Expecting time data type"
		# dateparser chokes on leading '+', thanks wikidata.
		return parser.parse(claim['value']['time'][1:])
	except:
		return datetime.today()


def get_url_name(string):
	string = unidecode(string.replace('.', '._').replace(' ', '_').replace('__', '_'))
	url = f"https://en.wikipedia.org/wiki/{string}"
	page = urlopen(url)
	soup = BeautifulSoup(page, 'html.parser')
	name = soup.find(id='firstHeading', class_='firstHeading').text
	return name.replace(' ', '_')


def compute_age(name, born_year):
	page = f'https://en.wikipedia.org/wiki/{get_url_name(name)}'
	# 1. use the wikipedia api to find the wikidata id for this page
	wikidata_id = id_for_page(page)
	# 2. use the wikidata id to get the birth and death dates
	span = lifespan_for_id(wikidata_id)

	return int(datetime.strftime(span, '%Y %d, %B')[:4]) - int(born_year[:4])
