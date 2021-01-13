import re
import spacy
import collections
import time
nlp = spacy.load('en_core_web_sm', parser = False, ner=False, textcat=False)
nlp.pipeline = [nlp.pipeline[0]]

def space_extend(matchobj):
	return ' ' + matchobj.group(0) + ' '
def pre_proc(text):
	text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/|\t', space_extend, text)
	text = text.strip(' \n')
	text = re.sub('\s+', ' ', text)
	return text

def _str(s):
	""" Convert PTB tokens to normal tokens """
	if (s.lower() == '-lrb-'):
		s = '('
	elif (s.lower() == '-rrb-'):
		s = ')'
	elif (s.lower() == '-lsb-'):
		s = '['
	elif (s.lower() == '-rsb-'):
		s = ']'
	elif (s.lower() == '-lcb-'):
		s = '{'
	elif (s.lower() == '-rcb-'):
		s = '}'
	return s

def process(parsed_text):
	output = {'word': [],
			'lemma': [],
			'pos': [],
			'pos_id': [],
			'ent': [],
			'ent_id': [],
			'offsets': [],
			'sentences': []}

	for token in parsed_text:
		#[(token.text,token.idx) for token in parsed_sentence]
		output['word'].append(_str(token.text))
		pos = token.tag_
		output['pos'].append(pos)

	return output


class HQE():
	def __init__(self, M, eta, R_topic, R_sub, pos_filter, verbose=False):
		self.M = M
		self.eta = eta
		self.R_topic = R_topic
		self.R_sub = R_sub
		self.pos_filter = pos_filter
		self.key_word_list = collections.defaultdict(list)
		self.subkey_word_list = collections.defaultdict(list)
		self.turn_id = -1
		self.time_consume = 0
		self.verbose = verbose
	def reset_history(self):
		if self.verbose:
			print('End session with {} turns, average query reformulation latency:{} secs'.format(self.turn_id+1, self.time_consume/(self.turn_id+1)))
			print('Clean keyword list for new session!')
		self.key_word_list = collections.defaultdict(list)
		self.subkey_word_list = collections.defaultdict(list)
		self.turn_id = -1
		self.time_consume = 0
	def rewrite(self, query, searcher):
		start_time = time.time()
		self.turn_id+=1
		self.key_word_extraction(query, searcher)
		if self.turn_id!=0:
			hits = searcher.search(query, 1)
			key_word = self.query_expansion(self.key_word_list, 0, self.turn_id)
			subkey_word = ''
			if not ( (hits[0].score > self.eta) ):
				end_turn = self.turn_id+1
				start_turn = end_turn - self.M
				if start_turn < 0:
					start_turn = 0
				subkey_word = self.query_expansion(self.subkey_word_list, start_turn, end_turn)
			query = key_word + ' ' + subkey_word + ' ' + query
		self.time_consume += time.time() - start_time
		return query
	def query_expansion(self, key_word_list, start_turn, end_turn):
		query_expansion = ''
		for turn in range(start_turn, end_turn+1):
			for word in key_word_list[turn]:
				query_expansion= query_expansion+' '+ word
		return query_expansion
	def key_word_extraction(self, query, searcher):
		proc_query = self.cal_word_score(query, searcher)
		# Extract topic keyword
		if self.pos_filter=='no':
			for i, word in enumerate(proc_query['word']):
				if ( (proc_query['score'][i]>=self.R_topic) ):
					self.key_word_list[self.turn_id].append(word)
				if ( (proc_query['score'][i]>=self.R_sub)&(proc_query['score'][i]<self.R_topic) ):
					self.subkey_word_list[self.turn_id].append(word)
		elif self.pos_filter=='pos':
			for i, word in enumerate(proc_query['word']):
				if(('NN' in proc_query['pos'][i]) or ('JJ' in proc_query['pos'][i]) ):
					if ( (proc_query['score'][i]>=self.R_topic) ):
						self.key_word_list[self.turn_id].append(word)
					if ( (proc_query['score'][i]>=self.R_sub)&(proc_query['score'][i]<self.R_topic) ):
						self.subkey_word_list[self.turn_id].append(word)
		elif self.pos_filter=='stp':
			for i, word in enumerate(proc_query['word']):
				if(word not in STOP_WORDS):
					if ( (proc_query['score'][i]>=self.R_topic) ):
						self.key_word_list[self.turn_id].append(word)
					if ( (proc_query['score'][i]>=self.R_sub)&(proc_query['score'][i]<self.R_topic) ):
						self.subkey_word_list[self.turn_id].append(word)
	def cal_word_score(self, query, searcher):
		nlp_query = nlp(pre_proc(query))
		proc_query = process(nlp_query)
		query_words = proc_query['word']
		proc_query['score'] = []

		for query_word in query_words:
			hits = searcher.search(query_word, 1)
			try:
				score = hits[0].score
				proc_query['score'].append(score)
			except:
				proc_query['score'].append(-1)

		return proc_query