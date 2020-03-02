import re
import matplotlib.pyplot as plt;
import json
plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

folderToFeatureTypeMapping = {
    "expo" : "ExponentialTimeGap",
    "expo_rlcache" : "RLCache_ExponentialTimeGap",
    "vanilla" : "Vanilla",
    "rlcache" : "RLCache"
}

folderToPlotText = {
    "expo" : "ExponentialTimeGaps",
    "expo_rlcache" : "ExponentialTimeGaps & RLCacheFeatures",
    "vanilla" : "Vanilla",
    "rlcache" : "RLCache"
}

with open('feature_mapping.json') as json_file:
    featureMapping = json.load(json_file)


def plot_features(feature, type):
    feature_list = sorted([(name, rval) for name, rval in feature.items()], key=lambda x: -x[1])
    feature_types = [name for name, rval in feature_list][:6]
    feature_values = [rval for name, rval in feature_list][:6]
    y_pos = np.arange(len(feature_types))

    plt.bar(y_pos, feature_values, align='center', alpha=0.5)
    plt.xticks(y_pos, feature_types)

    plt.title('Feature Ranks')
    plt.savefig('final/feature_imp.png')


def main():
    feature = dict()
    for i in range(5):
        filepath = "old-logs/expo-lru/booster_data/booster_" + str(i) + ".data"
        print(i)
        with open(filepath, 'r') as f:
            for line in f:
                search = re.search("^Column_[0-9]*=[0-9]*$", line)
                if search:
                    name, rval = search.group(0).split("=")
                    if name not in feature:
                        feature[name] = []
                    feature[name].append(int(rval))

    for key in feature.keys():
        print(featureMapping[folderToFeatureTypeMapping['expo']][key],feature[key])

    plt.clf()
    legend = []
    for k in list(feature.keys())[:5]:
            lista = [i for i in range(1, len(feature[k]) + 1)]
            plt.plot(lista, feature[k])
            y_pos = np.arange(len(lista)+1)[1:]
            print(y_pos)
            plt.xticks(y_pos, y_pos)
            legend.append(featureMapping[folderToFeatureTypeMapping['expo']][k])
    plt.legend(legend)
    plt.title("Feature Importance v/s Training iterations for \nfeatures with Exponential TimeGaps")
    # plt.grid()
    plt.xlabel("Training Iterations")
    plt.ylabel("Feature Importance")
    # plt.show()
    plt.savefig('FeatureImportanceVsIterations-Expo.png')


    # plot_features(feature, "Exponential Time Gaps")


if __name__ == "__main__":
    main()
