import re
import unicodedata
import MeCab
from tqdm import tqdm
import pickle
import pandas as pd
def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s
def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s
def cleanning(title):
    
    def regex(text):
        text_tmp = text
        text_tmp = re.sub('#.*', '', text_tmp)
        text_tmp = re.sub('\n', '', text_tmp)
        text_tmp = re.sub('@.*|\.*|\([A-Za-z0-9_]+\)', '', text_tmp)
        text_tmp = re.sub("[|・|！|？|!|?|･|\|＼|〇|♡|…|〜|~|\u3000|BR|:|お|ご]","",text_tmp)
        text_tmp = re.sub("[w|ｗ]","",text_tmp)
        text_tmp = re.sub('".*',"",text_tmp)
        text_tmp = re.sub(r'[!-/:-@[-`{-~]', r' ', text_tmp)
        text_tmp = re.sub(u'[■-♯]', ' ', text_tmp)
        text_tmp = re.sub("[「|」|『|』]","",text_tmp)
        text_tmp = re.sub(r'[0-9]+', "0", text_tmp)
        text_tmp = re.sub(r'\([^あ-ん\u30A1-\u30F4\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]+?\)',"",text_tmp)
        #text_tmp = text_tmp.replace("／","/")
        text_tmp = text_tmp.strip()
        return text_tmp
    
    new_title = []
    for t in title:
        new_title.append(regex(normalize_neologd(t)))
    return new_title
def remove_stopwords(sentence, stopwords):
    wakati = []
    for words in sentence:
        words = [word for word in words if word not in stopwords]
        wakati.append(words)
    return wakati
def get_stopwords():
    w = []
    with open("./model/json_dict/stopword.txt","r") as f:
        for i in f:
            w.append(i[:-1])
    return w
def get_dict(sentence):
    frq = {}
    for ww in sentence:
        for w in ww:
            if w not in frq.keys():
                frq[w] = 0
            frq[w] += 1
        
    return dict(sorted(frq.items(), key = lambda x:-x[1]))
def remove_mostwords(sentence,dic,n = 500):
    remove_word = []
    for w,v in dic.items():
        if v <= n:
            break
        remove_word.append(w)
    
    wakati = []
    for words in sentence:
        words = [word for word in words if word not in remove_word]
        wakati.append(words)
    return wakati

    
def tfidf_transformer(te_,name):
    from sklearn.feature_extraction.text import TfidfVectorizer
    with open(f'./model/pickle/{name}_tfidf.pickle', mode='rb') as fp:
        vectorizer = pickle.load(fp)
    tf2 = vectorizer.transform(te_).toarray()
    return tf2

def meisi_extractor(clean_data):
    l = []
    tagger = MeCab.Tagger("-Ochasen")
    for text in tqdm(clean_data):
        str_output = [line for line in tagger.parse(text).splitlines()
                   if "名詞" in line.split()[-1]]

        l.append(set([re.sub("\t.*","",s) for s in str_output]))
    return l

def _oubo(data):
    clean_data = cleanning(data["応募資格"])
    meisi_list = meisi_extractor(clean_data)
    stopword = get_stopwords()
    dic = get_dict(meisi_list)
    sentence = remove_stopwords(meisi_list,stopword)
    sentence = [" ".join(s) for s in sentence]
    return sentence
def _haizoku(data):
    clean_data = cleanning(data["（派遣先）配属先部署"])
    tagger = MeCab.Tagger("-Ochasen")
    haizoku = []
    for text in clean_data:
        str_output = [line for line in tagger.parse(text).splitlines()
                   if "名詞" in line.split()[-1]]

        haizoku.append([re.sub("\t.*","",s) for s in str_output])
    stopword = get_stopwords()
    dic = get_dict(haizoku)
    sentence = remove_stopwords(haizoku,stopword)
    sentence = [" ".join(s) for s in sentence]
    return sentence