from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

def parse_data(filepath):

    i = 0
    df = pd.DataFrame()

    with open(filepath, 'r') as f:
        for line in f:
            if "Silhouette coefficient" in line:
                df.at[i, "model"] = filepath.split('/')[-1].split('_')[0]
                df.at[i, "dataset"] = filepath.split('_')[1].split('.')[0]
                df.at[i, "k"] = int(i + 2)
                try:
                    df.at[i, "silhouette"] = float(line.strip().split(':')[1])
                except:
                    df.at[i, "silhouette"] = None
            elif "Calinski-Harabasz index" in line:
                try:
                    df.at[i, "ch_index"] = float(line.strip().split(':')[1])
                    i+=1
                except:
                    df.at[i, "ch_index"] = None
            else:
                pass
    
    return df

def plot_graph(df1, df2, dataset, score):


    ax = plt.figure().gca()
    ax.plot('k', score, data = df1)
    ax.plot('k', score, data = df2)
    ax.legend(['LayoutLM', 'LayoutLMv2'])
    plt.xlabel('k')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if score == 'silhouette':
        plt.ylabel('Silhouette Score')
        ax.set_title('Comparison of Models Using Silhouette Score\n'+dataset.upper())
    else:
        plt.ylabel('Calinski-Harabasx Index')
        ax.set_title('Comparison of Models Using Calinski-Harabasx Index\n'+dataset.upper())

    plt.savefig('plots/' + dataset + '_' + score + '_plot.png', dpi=300)
    plt.show()
    return

if __name__ == "__main__":

    lmv1_rc = parse_data("results/ktest/lmv1_rvl-cdip.txt")
    lmv2_rc = parse_data("results/ktest/lmv2_rvl-cdip.txt")

    plot_graph(lmv1_rc, lmv2_rc, 'rvl-cdip', 'silhouette')
    plot_graph(lmv1_rc, lmv2_rc, 'rvl-cdip', 'ch_index')

    lmv1_sr = parse_data("results/ktest/lmv1_sroie2019.txt")
    lmv2_sr = parse_data("results/ktest/lmv2_sroie2019.txt")
    plot_graph(lmv1_sr, lmv2_sr, 'sroie2019', 'silhouette')
    plot_graph(lmv1_sr, lmv2_sr, 'sroie2019', 'ch_index')

    

