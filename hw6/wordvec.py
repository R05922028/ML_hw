import jieba
from gensim.models import Word2Vec
jieba.set_dictionary('../hw6_data/dict.txt.big')

sents = []
f = open('../hw6_data/all_sents.txt', 'r')
lines = f.readlines()
for line in lines:
  slt = line.split('\n')[0]
  s_list = jieba.cut(slt)
  sent=[]
  for j in s_list:
    sent.append(j)
  sents.append(sent)  
#print(sents)

model = Word2Vec(sents, size = 100, alpha=0.025, window=15, min_count=3001, workers=8)
model.save('model.bin')
