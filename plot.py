import pandas as pd
import matplotlib.pylab as plt

def plotNumberOfClasses():
    df = pd.read_csv('mtsamples-expanded-min.csv')
    df = df.dropna()

    list = []
    for index, row in df.iterrows():
        list.append(row["medical_specialty"])

    listUnique = []

    for item in list:
        if listUnique.__contains__(item):
            continue
        else:
            listUnique.append(item)

    dictionary = dict()

    for item in listUnique:
        dictionary[item] = 0

    for item in list:
        dictionary[item] = dictionary[item] + 1

    lists = sorted(dictionary, key=dictionary.get, reverse=True)

    dictionarySort = dict()
    x = []
    y = []

    for item in lists:
        dictionarySort[item] = dictionary[item]
        y.append(dictionary[item])
        x.append(item)

    barlist = plt.bar(x, y)
    plt.xticks(rotation=-70)
    plt.show()

plotNumberOfClasses()