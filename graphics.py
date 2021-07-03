import pandas as pd
import matplotlib.pylab as plt

def plotNumberOfMoviesByGenre():
    df = pd.read_csv('dataset\mpst_full_data.csv')
    list = []
    for index, row in df.iterrows():
        list.append(row["tags"])

    listSplited = []

    for item in list:
        x = item.split(", ")
        for i in x:
            listSplited.append(i)

    listUnique = []

    for item in listSplited:
        if listUnique.__contains__(item):
            continue
        else:
            listUnique.append(item)

    dictionary = dict()

    for item in listUnique:
        dictionary[item] = 0

    for item in listSplited:
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
    plt.xticks(rotation=90)
    plt.show()

def plotNumberOfGenresByDescription():
    df = pd.read_csv('dataset\mpst_full_data.csv')
    list = []
    for index, row in df.iterrows():
        list.append(row["tags"])

    numbers = []

    for item in list:
        x = item.split(", ")
        numbers.append(len(x))

    dictionary = dict()

    for item in range(max(numbers)+1):
        dictionary[item] = 0

    for item in numbers:
        dictionary[item] = dictionary[item] + 1

    lists = dictionary

    dictionarySort = dict()
    x = []
    y = []

    for item in lists:
        dictionarySort[item] = dictionary[item]
        y.append(dictionary[item])
        x.append(item)

    plt.bar(x, y)
    plt.xticks(rotation=90)
    plt.show()


if __name__ == '__main__':
    plotNumberOfMoviesByGenre()
    plotNumberOfGenresByDescription()