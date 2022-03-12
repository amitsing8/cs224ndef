import json
import os
import copy
from BackTranslation import BackTranslation

indir = 'datasets/oodomain_train/'
outdir = 'datasets/oodomain_augtrain/'


def translate(srcString, lang='es'):
    trans = BackTranslation(url=['translate.google.com', 'translate.google.co.kr',
                                 ], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})
    result = trans.translate(srcString, src='en', tmp=lang)
    return result.result_text


def getTranlateDict(ctx, answerIndices):
    dictOfAnswers = {}
    nextIndex = 0
    for k, v in enumerate(ctx):
        if v == ".":
            if len(answerIndices) > 0:
                if answerIndices[0] >= nextIndex and answerIndices[0] <= k-1:
                    dictOfAnswers[(nextIndex, k-1)] = False
                    answerIndices.remove(answerIndices[0])
                else:
                    dictOfAnswers[(nextIndex, k-1)] = True
                nextIndex = k
    return dictOfAnswers


def getTuples(answerIndicesList, ctxlen):
    ctx_tuples = []
    ans_tuples = []
    ctxtuplenow = True
    startIdx = 0
    previdx = None
    for idx in answerIndicesList:
        if idx == 0:
            ctxtuplenow = False
            previdx = idx
            continue
        if previdx is not None and idx == previdx+1:
            previdx = idx
            continue
        if ctxtuplenow:
            ctx_tuples.append((startIdx, idx))
            startIdx = idx
            previdx = idx
            ctxtuplenow = False
        else:
            ans_tuples.append((startIdx, previdx))
            ctx_tuples.append((previdx, idx))
            startIdx = idx
            previdx = idx
    if previdx is None:
        ctx_tuples.append((0, ctxlen))
    else:
        if (previdx < ctxlen-1):
            ans_tuples.append((startIdx, previdx))
            ctx_tuples.append((previdx, ctxlen))
        else:
            ans_tuples.append((startIdx, previdx))
    return ctx_tuples, ans_tuples


def translate_non_answers(ctx, answerIndicesList, lang='es'):
    xlt_ctx = ""
    ctxtuplenow = True
    startIdx = 0
    previdx = None
    for idx in answerIndicesList:
        if idx == 0:
            ctxtuplenow = False
            previdx = idx
            continue
        if previdx is not None and idx == previdx+1:
            previdx = idx
            continue
        if ctxtuplenow:
            # ctx_tuples.append((startIdx, idx))
            if idx - startIdx > 10:
                xlt_ctx += translate(ctx[startIdx:idx], lang) + " "
            else:
                xlt_ctx += ctx[startIdx:idx]
            startIdx = idx
            previdx = idx
            ctxtuplenow = False
        else:
            # ans_tuples.append((startIdx, previdx))
            xlt_ctx += ctx[startIdx:previdx+1]
            try:
                if idx - previdx > 10:
                    xlt_ctx += " " + translate(ctx[previdx+1:idx], lang) + " "
                else:
                    xlt_ctx += ctx[previdx+1:idx]
            except:
                print(previdx, idx, len(ctx))
                exit(0)
            # ctx_tuples.append((previdx, idx))
            startIdx = idx
            previdx = idx
    if previdx is None:
        ctx_tuples.append((0, ctxlen))
        xlt_ctx += translate(ctx, lang)
    else:
        if (previdx < len(ctx)-1):
            # ans_tuples.append((startIdx, previdx))
            xlt_ctx += ctx[startIdx:previdx+1]
            # ctx_tuples.append((previdx, ctxlen))
            if idx - previdx > 10:
                xlt_ctx += " " + translate(ctx[previdx+1:], lang)
            else:
                xlt_ctx += ctx[previdx+1:]
        else:
            # ans_tuples.append((startIdx, previdx))
            xlt_ctx += ctx[startIdx:previdx+1]
    return xlt_ctx


def translate_line_by_line(ctx, answerIndices, lang='es'):
    newCtx = ""
    dictOfAnswers = getTranlateDict(ctx, answerIndices)
    for k, v in dictOfAnswers:
        if not dictOfAnswers[(k, v)]:
            newCtx += ctx[k:v]
        else:
            newCtx += translate(ctx[k:v])
    return newCtx


def main():
    mode = "qas_translate"
    langList = ['fr', 'es']
    data_files = ['duorc', 'race', 'relation_extraction']
    mode = "ctx_translate"
    # langList = ['es']
    # data_files = ['duorc']
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for file in data_files:
        fr = open(indir+file)
        fw = open(outdir+file, 'w')
        dat = fr.read()
        jdat = json.loads(dat)
        jdat2 = copy.deepcopy(jdat)
        for i1, entry in enumerate(jdat['data']):
            for i2, para in enumerate(entry['paragraphs']):
                answerIndicesSet = set([])
                answerIndices = []
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
                            answerIndicesSet.update(list(range
                                                         (answer['answer_start'],
                                                          answer['answer_start']+len(answer['text']))))

                answerIndicesList = sorted(answerIndicesSet)
                # ctx_tuples, ans_tuples=getTuples(
                #     answerIndicesList, len(para['context']))

                answerIndices = sorted(set(answerIndices))
                if mode == "ctx_translate":
                    newc = {}
                    for lang in langList:
                        xltpara = translate_non_answers(
                            para['context'], answerIndicesList, lang)
                        newqas = copy.deepcopy(para['qas'])
                        for el in newqas:
                            for ans in el["answers"]:
                                ans["answer_start"] = xltpara.index(
                                    ans["text"])
                        newc['qas'] = newqas
                        # newc['context']=translate_line_by_line(
                        #     para['context'], answerIndices, lang)
                        newc['context'] = xltpara

                        jdat2['data'][i1]['paragraphs'].append(newc)

        odat = json.dumps(jdat2)
        fw.write(odat)


if __name__ == '__main__':
    main()
