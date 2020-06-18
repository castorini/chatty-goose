import re
import spacy

nlp = spacy.load('en', parser = False)

def space_extend( matchobj):
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
        # output['pos_id'].append(token2id(pos, POS, 0))

        ent = 'O' if token.ent_iob_ == 'O' else (token.ent_iob_ + '-' + token.ent_type_)
        output['ent'].append(ent)
        # output['ent_id'].append(token2id(ent, ENT, 0))

        output['lemma'].append(token.lemma_ if token.lemma_ != '-PRON-' else token.text.lower())
        output['offsets'].append((token.idx, token.idx + len(token.text)))

    word_idx = 0
    for sent in parsed_text.sents:
        output['sentences'].append((word_idx, word_idx + len(sent)))
        word_idx += len(sent)

    assert word_idx == len(output['word'])
    return output

def word_score(query, searcher, JString):
    nlp_query = nlp(pre_proc(query))
    proc_query = process(nlp_query)
    query_words = proc_query['word']
    proc_query['score'] = []

    for query_word in query_words:
        hits = searcher.search(JString(query_word.encode('utf8')), 1)
        try:
            score = hits[0].score
            proc_query['score'].append(score)
        except:
            proc_query['score'].append(-1)

    return proc_query

def read_corpus(corpus_path):
    id_to_doc = {}
    with open(corpus_path, 'r') as f:
        print('Read Corpus...')
        for line in f:
            try:
                text = line.strip().split("\t")
                docid = text[0]
                doc= ' '.join(text[1:])
                id_to_doc[docid] = doc
            except:
                print("skip %s"%(line.strip()))

    return id_to_doc


def exact_key_word(proc_query, turn_id, low, up, key_word_list, filter):
    if filter=='no':
        for i, word in enumerate(proc_query['word']):
            if ( (proc_query['score'][i]>=(low))&(proc_query['score'][i]<up)   ):
                key_word_list[turn_id].append(word)
    elif filter=='pos':
        for i, word in enumerate(proc_query['word']):
            if(('NN' in proc_query['pos'][i]) or ('JJ' in proc_query['pos'][i]) ):
                if ( (proc_query['score'][i]>=(low))&(proc_query['score'][i]<up)   ):
                    key_word_list[turn_id].append(word)
    elif filter=='stp':
        for i, word in enumerate(proc_query['word']):
            if(word not in STOP_WORDS):
                if ( (proc_query['score'][i]>=(low))&(proc_query['score'][i]<up)   ):
                    key_word_list[turn_id].append(word)
    return key_word_list

def query_expansion(key_word_list, start_turn, end_turn):
    query_expansion = ''
    for turn in range(start_turn, end_turn):
        for word in key_word_list[turn]:
            query_expansion= query_expansion+' '+ word
    return query_expansion