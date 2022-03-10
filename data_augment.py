import json
import copy
from BackTranslation import BackTranslation

indir = 'datasets/oodomain_train/'
outdir = 'datasets/oodomain_augtrain/'

def translate(srcString, lang='es'):
  trans = BackTranslation(url=['translate.google.com', 'translate.google.co.kr',
                                ], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})
  result = trans.translate(srcString, src='en', tmp=lang)
  return result.result_text

def getTranlateDict (ctx, answerIndices):
  dictOfAnswers = {}
  nextIndex = 0
  for  k,v in enumerate(ctx):
    if v == ".":
      if len(answerIndices) > 0:
        if answerIndices[0]>=nextIndex and answerIndices[0]<=k-1:
          dictOfAnswers[(nextIndex,k-1)] = False
          answerIndices.remove(answerIndices[0])
        else:
          dictOfAnswers[(nextIndex,k-1)] = True
        nextIndex = k
  return dictOfAnswers

def translate_line_by_line(ctx, answerIndices, lang='es'):
  newCtx = ""
  dictOfAnswers = getTranlateDict (ctx, answerIndices)
  for k, v in dictOfAnswers:
    if not dictOfAnswers[(k,v)]:
      newCtx += ctx[k:v]
    else:
      newCtx += translate(ctx[k:v])
  return newCtx

def main():
    answerIndices = []
    mode = "qas_translate"
    langList = ['fr', 'es']
    data_files = ['duorc', 'race', 'relation_extraction']
    #mode = "ctx_translate"
    #langList = ['es']
    #data_files = ['duorc']

    for file in data_files:
        fr = open(indir+file)
        fw = open(outdir+file, 'w')
        dat = fr.read()
        jdat = json.loads(dat)
        jdat2 = copy.deepcopy(jdat)
        for i1, entry in enumerate(jdat['data']):
            for i2, para in enumerate(entry['paragraphs']):
                for i3, el in enumerate(para['qas']):
                    if mode == "qas_translate":
                        newq = {}
                        for lang in langList:
                            newq['question'] = translate(el['question'], lang)
                            newq['answers'] = el['answers']
                            newq['id'] = el['id']
                        jdat2['data'][i1]['paragraphs'][i2]['qas'].append(newq)
                    else:
                        for answer in el['answers']:
                            answerIndices.append(answer['answer_start'])
                            
                answerIndices = sorted(set(answerIndices))
                if mode == "ctx_translate":
                    newc = {}
                    newc['qas'] = para['qas']
                    newc['context'] = translate_line_by_line(para['context'], answerIndices, lang)
                    jdat2['data'][i1]['paragraphs'].append(newc)

        odat = json.dumps(jdat2)
        fw.write(odat)

