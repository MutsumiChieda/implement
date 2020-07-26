# naive bayes implementation

import numpy as np

documents_raw = ["good bad good good",
    "exciting exciting",
    "good good exciting boring",
    "bad boring boring boring",
    "bad good bad",
    "bad bad boring exciting"]

# 0: positive, 1: negative
teacher = [0,0,0,1,1,1]

test = "good good bad boring".split()

# reshape
documents = []
for document in documents_raw:
    documents.append(document.split())
n_samples = len(documents)

# flatten and extract unique vocablary
vocabulary = sorted(set(sum(documents, [])))
n_vocabulary = len(vocabulary)

# count samples
n_class = len(set(teacher))
n_c = [0] * n_class
n_cw = np.zeros((n_class, n_vocabulary))

for doc_i, document in enumerate(documents):
    class_ = teacher[doc_i]
    n_c[class_] += 1
    for voc_i, word in enumerate(vocabulary):
        n_cw[class_][voc_i] += 1 if(word in document) else 0

# possiblility
p_c = [0] * n_class
p_cw = np.zeros((n_class, n_vocabulary))
for class_ in range(n_class):
    p_c[class_] = n_c[class_] / n_samples
    for voc_i, word in enumerate(vocabulary):
        p_cw[class_][voc_i] = n_cw[class_][voc_i] / n_c[class_]

# predict
pred_proba = [1] * n_class
for class_ in range(n_class):
    pred_proba[class_] *= p_c[class_]
    for voc_i, word in enumerate(vocabulary):
        pred_proba[class_] *= p_cw[class_][voc_i] if(word in test) \
            else (1 - p_cw[class_][voc_i])

print('predicted probability')
print('positive, negative')
print(pred_proba)