import json
import re
import os
import pickle
import matplotlib.pyplot as plt;
import numpy as np
import random

folderToFeatureTypeMapping = {
    "expo" : "ExponentialTimeGap",
    "expo_rlcache" : "RLCache_ExponentialTimeGap",
    "vanilla" : "Vanilla",
    "rlcache" : "RLCache"
}

folderToPlotText = {
    "expo" : "Features(ExponentialTimeGaps)",
    "expo_rlcache" : "Features(ExponentialTimeGaps & RLCache)",
    "vanilla" : "Features(Vanilla)",
    "rlcache" : "Features(RLCache)"
}

with open('feature_mapping.json') as json_file:
    featureMapping = json.load(json_file)

featureImportanceGlobal = {}
ohrGlobal = {}
bhrGlobal = {}
colorMapping = {}
chunkOhrGlobal = {}
chunkBhrGlobal = {}

def checkEquality(x,y, objName):
    shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
    print("{} is equal : {}".format(objName,len(x) == len(shared_items)))

def loadObj(fileName):
    with open(fileName, 'rb') as handle:
        objLoaded = pickle.load(handle)
    return objLoaded

def saveObj(obj, fileName, objName):
    with open(fileName, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(fileName, 'rb') as handle:
        objLoaded = pickle.load(handle)

    checkEquality(obj, objLoaded, objName)


def printFeatureImportance(folderName, featureType):
    # open booster file
    filepath = "./logs/" + folderName + "/booster_data/booster_0.data"

    featureImportanceList = []

    print("Feature Importance\n")
    with open(filepath, 'r') as f:
        for line in f:
            search = re.search("^Column_[0-9]*=[0-9]*$", line)
            if search:
                name, rval = search.group(0).split("=")
                featureName = featureMapping[featureType][name]
                featureImportanceList.append((featureName, rval))
                print("[+]  {} : {}".format(featureName, rval))
    featureImportanceGlobal[folderName] = featureImportanceList
    print("\n")

'''
if (webcache->lookup(&req)) {
            hit++;
            outfile << id << ' ' << size << ' ' << '1' << ' ' << '1' << std::endl;
        } else {
            req.setFeatureVector(prev_feature);
            webcache->admit(&req);
            if (webcache->lookup(&req)) {
                outfile << id << ' ' << size << ' ' << '1' << ' ' << '0' << std::endl;
            } else {
                outfile << id << ' ' << size << ' ' << '0' << ' ' << '0' << std::endl;
            }
        }
'''

totalCount = 10000000
chunkCount = 1000000



def printMetricsPerChunk(folderName):
    filepath = "./logs/" + folderName + "/cache_decisions.out"
    bytesRequested = 0
    objectsRequested = 0
    bytesHit = 0
    objectsHit = 0

    count = 0
    secCount = 0
    chunkNo = 0

    bhrList = []
    ohrList = []

    with open(filepath, 'r') as f:
        for line in f:

            if count > chunkCount:
                chunkNo+=1
                OHR = objectsHit * 1.0 / objectsRequested
                BHR = bytesHit * 1.0 / bytesRequested
                print("[+] Chunk: {}".format(chunkNo))
                print("[+]      OHR : {}".format(OHR))
                print("[+]      BHR : {}".format(BHR))
                ohrList.append(OHR)
                bhrList.append(BHR)

                bytesRequested = 0
                objectsRequested = 0
                bytesHit = 0
                objectsHit = 0
                count = 0

            if secCount > totalCount:
                break

            id, size, found, existed = line.split(' ')
            objectsRequested += 1
            bytesRequested += int(size)
            if int(existed) == 1:
                bytesHit += int(size)
                objectsHit += 1

            count += 1
            secCount+=1

    chunkOhrGlobal[folderName] = ohrList
    chunkBhrGlobal[folderName] = bhrList

    print("\n")


def printMetrics(folderName):
    filepath = "./logs/" + folderName + "/cache_decisions.out"
    bytesRequested = 0
    objectsRequested = 0
    bytesHit = 0
    objectsHit = 0

    count = 0
    with open(filepath, 'r') as f:
        for line in f:

            if count > totalCount:
                break

            id, size, found, existed = line.split(' ')
            objectsRequested += 1
            bytesRequested += int(size)
            if int(existed) == 1:
                bytesHit += int(size)
                objectsHit += 1

            count+=1

    OHR = objectsHit * 1.0 / objectsRequested
    BHR = bytesHit * 1.0 / bytesRequested
    ohrGlobal[folderName] = OHR
    bhrGlobal[folderName] = BHR

    print("Statistics")

    print("[+] OHR : {}".format(OHR))
    print("[+] BHR : {}".format(BHR))

    print("\n")


#parseFolder
def processFolder(folderName):
    cacheSize, featureType, startingCacheAlgorithm = folderName.split("-")
    featureType = folderToFeatureTypeMapping[featureType]
    print("*" * 20)
    print("Processing folder {}".format(folderName))
    print("\nFeature Type : {}, Starting Cache Algorithm : {}, Cache Size : {}\n".format(featureType, startingCacheAlgorithm, cacheSize))
    printFeatureImportance(folderName, featureType)
    printMetrics(folderName)
    printMetricsPerChunk(folderName)
    print("*" * 20)

def getPlotText(folderName):
    cacheSize, featureType, startingCacheAlgorithm = folderName.split("-")
    return "{}, Start({})".format(folderToPlotText[featureType],startingCacheAlgorithm.upper())


def plotHorizontal(dictionary, title, computeLabel=True,limit=200):
    global colorMapping
    plt.clf()

    feature_list = sorted([(name, rval) for name, rval in dictionary.items()], key=lambda x: x[1])
    if computeLabel:
        feature_types = [getPlotText(name) for name, rval in feature_list]  # [:6]
    else:
        feature_types = [name for name, rval in feature_list]  # [:6]
    feature_values = [rval for name, rval in feature_list]  # [:6]
    colors = [colorMapping[name] for name, rval in feature_list]  # [:6]

    fig, ax = plt.subplots()

    y_pos = np.arange(len(feature_types))

    bar_plot = ax.barh(y_pos, feature_values, align='center', color = colors)

    ax.set_title(title)
    bar_label = feature_types
    ax.axes.get_yaxis().set_ticks([])

    font = {'family': 'monospace',
            'size': 8}

    plt.rc('font', **font)

    for idx, rect in enumerate(bar_plot):
        ax.text(rect.get_x(), (rect.get_y() + rect.get_width() ),
                bar_label[idx],
                color='black', fontweight='bold',
                ha='left', va='center')

    plt.savefig('{}.png'.format(title))


def plot(dictionary, title):
    global colorMapping

    feature_list = sorted([(name, rval) for name, rval in dictionary.items()], key=lambda x: -x[1])
    feature_types = [getPlotText(name) for name, rval in feature_list]#[:6]
    feature_values = [rval for name, rval in feature_list]#[:6]
    colors = [colorMapping[name] for name, rval in feature_list]#[:6]
    y_pos = np.arange(len(feature_types))

    fig, ax = plt.subplots()
    bar_label = feature_types
    bar_plot = plt.bar(y_pos, feature_values,width=1.0, alpha=0.5, color = colors)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    for idx, rect in enumerate(bar_plot):
        ax.text(rect.get_x() + rect.get_width() / 2., 0,
                bar_label[idx],
                ha='center', va='bottom', rotation=90)

    plt.title(title)
    plt.savefig('tmp.png')

def getRandomColors(len):
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len)]
    return color

def initColor(folderNames):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(len(folderNames))]
    for f,c in zip(folderNames,color):
        colorMapping[f] = c

def getStatsGroupedByKey(key):
    folderNames = [x for x in os.listdir("logs") if key in x]

    #plotting BHR
    newDict = {}
    for f in folderNames:
        newDict[f] = chunkBhrGlobal[f]

    plotLine(newDict, "Byte Hit Ratio","Byte Hit Ratio for {}".format(folderToPlotText[key.replace("-","")]), key.replace("-",""))

    # plotting OHR
    newDict = {}
    for f in folderNames:
        newDict[f] = chunkOhrGlobal[f]

    plotLine(newDict, "Object Hit Ratio", "Object Hit Ratio for {}".format(folderToPlotText[key.replace("-","")]), key.replace("-",""))

    '''
    #find max ohr, print horizontal for its features
    newDict = {}
    for f in folderNames:
        newDict[f] = ohrGlobal[f]
    top = sorted([(name,rval) for name, rval in newDict.items()], key=lambda x: -x[1])[0]

    # newDict = {}
    topFeatureImportance = featureImportanceGlobal[top[0]]
    print(topFeatureImportance)
    feature_types = [name for (name,rval) in topFeatureImportance][:10]
    feature_values = [rval for (name,rval) in topFeatureImportance][:10]
    print(feature_types)
    print(feature_values)
    plt.clf()

    fig, ax = plt.subplots()
    bar_label = feature_types
    y_pos = np.arange(len(feature_types))
    bar_plot = plt.bar(y_pos, feature_values, width=1.0, alpha=0.5,color = getRandomColors(len(feature_types)))
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    for idx, rect in enumerate(bar_plot):
        ax.text(rect.get_x() + rect.get_width() / 2., 0,
                bar_label[idx],
                ha='center', va='bottom', rotation=90)

    plt.title('{}.png'.format("Feature_Importance_{}".format(getPlotText(top[0]))))
    plt.savefig('{}.png'.format("Feature_Importance_{}".format(getPlotText(top[0]))))
    '''





def plotLine(dictionary, metric, title, groupBy):
    plt.clf()
    legend = []
    for k in dictionary.keys():
        lista = [i for i in range(1,len(dictionary[k])+1)]
        plt.plot(lista, dictionary[k])
        legend.append(getPlotText(k))
        # break

    plt.ylabel(metric)
    plt.xlabel('Iteration')
    plt.legend(legend,prop={"size":8})
    # plt.grid()
    plt.title(title)
    if metric == 'Byte Hit Ratio':
        plt.savefig('line_{}_{}.png'.format("BHR",groupBy.upper()))
    else:
        plt.savefig('line_{}_{}.png'.format("OHR", groupBy.upper()))

def main(process):
    global colorMapping, ohrGlobal, bhrGlobal, featureImportanceGlobal, chunkBhrGlobal, chunkOhrGlobal
    if process:
        folderNames = [x for x in os.listdir("logs") if x.startswith("16gb")]
        initColor(folderNames)
        saveObj(colorMapping, "colorMapping.pkl", "colorMapping")
        for f in folderNames:
            processFolder(f)
        saveObj(ohrGlobal, "ohrGlobal.pkl", "ohrGlobal")
        saveObj(bhrGlobal, "bhrGlobal.pkl", "bhrGlobal")
        saveObj(featureImportanceGlobal, "featureImportanceGlobal.pkl", "featureImportanceGlobal")
        saveObj(chunkBhrGlobal, "chunkBhrGlobal.pkl", "chunkBhrGlobal")
        saveObj(chunkOhrGlobal, "chunkOhrGlobal.pkl", "chunkOhrGlobal")
    else:
        ohrGlobal = loadObj("ohrGlobal.pkl")
        bhrGlobal = loadObj("bhrGlobal.pkl")
        chunkBhrGlobal = loadObj("chunkBhrGlobal.pkl")
        chunkOhrGlobal = loadObj("chunkOhrGlobal.pkl")
        featureImportanceGlobal = loadObj("featureImportanceGlobal.pkl")
        colorMapping = loadObj("colorMapping.pkl")
        # plotHorizontal(ohrGlobal,'Object Hit Ratio')
        # plotHorizontal(bhrGlobal, 'Byte Hit Ratio')

        # getStatsGroupedByKey("lru")

        # getStatsGroupedByKey("gdsf")

        # getStatsGroupedByKey("lfuda")

        # getStatsGroupedByKey("adaptsize")

        # getStatsGroupedByKey("-vanilla-")

        # getStatsGroupedByKey("-expo-")

        # getStatsGroupedByKey("-expo_rlcache-")

        getStatsGroupedByKey("-rlcache-")


if __name__ == "__main__":
    main(False)