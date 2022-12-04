import pandas as pd
import numpy as np

df = pd.read_csv('heart_train_data.csv')

outputDic = {}

xd = df.drop('target', axis=1)

for yValue in df['target'].unique():
    Ic = df['target']==yValue
    tempDic = {}
    for item in xd:
        dic = {}
        for el in xd[item].unique():
            dic[el] = sum(xd[item][Ic]==el)/len(xd[item][Ic])
        tempDic[item] = dic
    outputDic[yValue] = tempDic
pyc = {0:1 - df['target'].mean(), 1:df['target'].mean()}

def predict(input, outputDic, pyc):
    dic = {}
    for item in outputDic:
        prod = pyc[item]
        for el in input:
            prod *= outputDic[item][el][input[el]]
        dic[item] = prod
    max = 0
    for item in dic:
        if max < dic[item]:
            max = dic[item]
            index = item
    return index

df1 = pd.read_csv('heart_validate_data.csv')
matrix = df1.values.tolist()
result = []
for item in matrix:
    pred = predict({"cp": item[0], "exang": item[1], "thal": item[2]}, outputDic, pyc)
    if pred == item[3]:
        result.append(1)
    else:
        result.append(0)

print('Question A')
print(outputDic[1]["thal"][2])
print("Question B")
print(sum(result)/len(result))
print("Question C")
print(df['target'].mean())
print("Question D")
print(predict({"cp":1,"exang":1,"thal":2}, outputDic, pyc))