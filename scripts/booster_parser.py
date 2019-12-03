import re
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

def plot_features(feature):
    feature_list = sorted([(name, rval) for name, rval in feature.items()], key=lambda x: -x[1])
    feature_types = [name for name, rval in feature_list][:6]
    feature_values = [rval for name, rval in feature_list][:6]
    y_pos = np.arange(len(feature_types))

    plt.bar(y_pos, feature_values, align='center', alpha=0.5)
    plt.xticks(y_pos, feature_types)
    
    plt.title('Feature Ranks')
    plt.savefig('feature_imp.png')

def main():
    feature = dict() 
    for i in range(5):
        filepath = "../booster_data/booster_" + str(i) + ".data"
        print(i)
        with open(filepath, 'r') as f:
            for line in f:
                search = re.search("^Column_[0-9]*=[0-9]*$", line)
                if search:
                    name, rval = search.group(0).split("=")
                    if name not in feature:
                        feature[name] = 0
                    feature[name] += int(rval)

    plot_features(feature)

if __name__ == "__main__":
    main()
