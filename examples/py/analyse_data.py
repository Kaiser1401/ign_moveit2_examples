import sys

import data_utils
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import math
import numpy as np
from math3d import (Transform, Orientation, PositionVector, Versor)
import classify
import random
from pathlib import Path


# TODO make this a lib!!
# chosse colors and styles, copy to latex and inkscape
# have look ata seaborn for plotting https://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot
# https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html

# ,'ieee','std-colors'
plt_style_normal = ['science','no-latex','nature']

plt.style.use(plt_style_normal)



def plt_style_scatter(bScatter):
    if bScatter:
        plt.style.use(plt_style_normal + ['scatter'])
    else:
        plt.style.use(plt_style_normal)


print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
#print(plt.rcParams['axes.prop_cycle'].by_key()['linestyle'])
#print(plt.rcParams['axes.prop_cycle'].by_key()['marker'])

def color(idx):
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    l = len(cycle)
    idx = idx % l
    return cycle[idx]



v_names = ['s_x', 's_y', 's_z', 's_roll', 's_pitch', 's_yaw']
v_names_offsets = ['o_x', 'o_y', 'o_z', 'o_roll', 'o_pitch', 'o_yaw']
o_names = ['success','handling_error']
f_names = ['sum_s_trans', 'sum_s_rot', 'max_s_trans', 'max_s_rot', 'o_distance',  'o_angle', 'prediction_rate']

tf_lookup = {True: 1.0, False: 0.0}

pd.set_option("display.precision", 2)

#TODO: colour palettes and symbols -> https://ranocha.de/blog/colors/#gsc.tab=0 + ask Sebastien? -> https://github.com/garrettj403/SciencePlots ?

#plt.xkcd()

# random.seed(1234)

bSaveFigures = False
figurefolder = ""

def figsave(plot_axis):
    if not bSaveFigures:
        return plot_axis
    p = Path(figurefolder)
    Path.mkdir(p, parents=True, exist_ok=True)
    i = 1
    fn = p / f'figure{i}.svg'
    fn_png = p / f'figure{i}.png'
    while fn.exists():
        i+=1
        fn = p / f'figure{i}.svg'

    plot_axis.figure.savefig(fn)
    plot_axis.figure.savefig(fn_png)

    return plot_axis




def confusion_data_binary(cm:classify.metrics.ConfusionMatrix, bPrint=False):
    # binary cm (TRUE, FALSE)
    tp = cm.true_positives(True)
    tn = cm.true_negatives(True)
    fp = cm.false_positives(True)
    fn = cm.false_negatives(True)

    #print(tp, tn, fp, fn)
    sum = tp+tn+fp+fn
    suc = tp + fn

    metrics = {
        'sum': sum,
        'succesfull interactions': suc,
        'correct predictions' : tp+tn,
        'accuracy': (tp+tn) / sum,
        'error': (fp + fn) / sum,
        'precision': tp/(tp+fp),
        'recall': tp/(tp+fn),
        'specificity': tn/(tn+fp),

    }

    decimals = 2
    if bPrint:
        rounded_metrics = {key : round(metrics[key],decimals) for key in metrics}
        print("------------------")
        print("      / Predicted")
        print("Act. /")
        print(cm)
        {print(f"{k:<12}: {v}") for k,v in rounded_metrics.items()}
        print("------------------")
        print("")

    return metrics




def relearn(data,bshuffle=False):


    def get_succ_prob(pred_prob_list):
        if len(pred_prob_list) <1 :
            return None
        if True in pred_prob_list:
            return pred_prob_list[True]
        else:
            return 1.0-pred_prob_list[False]


    i_stop_learning_after = 0
    clf = classify.Classifyer()
    clf.resetConfusion()

    predicted_probability__iter = []
    result__iter = []
    predicted_probability__post = []
    result__post = []

    if bshuffle > 0:
        r_seed = random.randrange(sys.maxsize)
        print(f'Randomized list with seed: {r_seed}')
        random.seed(r_seed)
        random.shuffle(data)

    learned_count = 0

    for e in data:
        assert isinstance(e, data_utils.DataEntry)
        pred = clf.predict(e.sampled_variance)
        pred_prob = clf.predict_prob(e.sampled_variance)
        outcome = e.b_outcome
        if learned_count < i_stop_learning_after or i_stop_learning_after < 1:
            clf.learn(e.sampled_variance, outcome)
            learned_count += 1
        clf.storeOutcome(pred, outcome)

        tmp = get_succ_prob(pred_prob)
        if tmp is not None:
            predicted_probability__iter.append(tmp)
            result__iter.append(tf_lookup[e.b_outcome])




    r = 100 # running window

    pred_ratio_iterative = clf.get_equal_ratios().copy()
    pred_iter_running = clf.get_equal_ratios(r).copy()

    print("Iterative:")
    confusion_data_binary(clf.confusion,True)



    clf.resetConfusion(bAndList=True)
    for e in data:
        assert isinstance(e, data_utils.DataEntry)
        pred = clf.predict(e.sampled_variance)
        pred_prob = clf.predict_prob(e.sampled_variance)
        outcome = e.b_outcome
        clf.storeOutcome(pred, outcome)

        predicted_probability__post.append(pred_prob[True])
        result__post.append(tf_lookup[e.b_outcome])

    pred_ratio_post = clf.get_equal_ratios().copy()
    pred_post_running = clf.get_equal_ratios(r).copy()

    print("Post:")
    confusion_data_binary(clf.confusion, True)

    #https://www.nosimpler.me/accuracy-precision/

    diff_iter_post_running = np.abs(pred_post_running-pred_iter_running) * 10

    pd_correct_prediction_ratios = pd.DataFrame(
        { 'pred_cum_ratio_iter': pred_ratio_iterative,
          'pred_cum_ratio_post': pred_ratio_post,
          f'pred_iter_r{r}': pred_iter_running,
          f'pred_post_r{r}': pred_post_running,
#          'difference_iter_post*10': diff_iter_post_running,
          }
    )

    figsave(pd_correct_prediction_ratios[['pred_cum_ratio_iter']].plot())

    pd_pred_probabilities_iter = pd.DataFrame(
        {
            'probability_predicted': predicted_probability__iter,
            'outcome': result__iter
        }
    )

    pd_pred_probabilities_post = pd.DataFrame(
        {
            'probability_predicted': predicted_probability__post,
            'outcome': result__post

        }
    )

    n_bins = 10
    bEqualBinSize = True

    pd_binned_probabilities = pd_pred_probabilities_iter.copy()
    bin_labels = []
    for b in range(n_bins):
        bin_labels.append(str(b+1))

    if bEqualBinSize:
        # equal bin count
        pd_binned_probabilities["interval"] = pd.qcut(pd_binned_probabilities['probability_predicted'],n_bins)
    else:
        # equal bin width
        pd_binned_probabilities["interval"]=pd.cut(pd_binned_probabilities['probability_predicted'],np.linspace(0,1, n_bins+1),right=True,include_lowest=True)

    pd_tmp = pd_binned_probabilities.groupby("interval").mean()
    pd_tmp["diff"] = pd_tmp["probability_predicted"] - pd_tmp["outcome"]
    pd_tmp['bins'] = bin_labels
    pd_tmp['n'] = pd_binned_probabilities.groupby("interval").count()['outcome']
    pd_tmp['s'] = pd_binned_probabilities.groupby("interval").sum()['outcome']
    pd_tmp['inter'] = pd_tmp.index

    pd_print = pd_tmp[['bins','inter','n','s','probability_predicted','outcome','diff']]
    print(pd_print.to_string(index=False))


    # figsave(pd_tmp.plot('bins', y=['probability_predicted', 'outcome', 'diff'], kind='bar', grid=True))

    figsave(pd_tmp.plot(y=['probability_predicted', 'outcome', 'diff'], kind='bar', grid=True))




def load_file(fn):
    data = data_utils.load_data(fn)
    return data

def data_2_pandas(data):
    if len (data) == 0:
        print("No Data")
        return

    n_input_vars = 6

    pdata_list = []

    for e in data:
        assert isinstance(e, data_utils.DataEntry)

        if e.b_outcome is None:
            continue

        tmpdict = {}
        # standard_deviation as numpy
        npstd = np.sqrt(np.array(e.sampled_variance))
        # actual offset as numpy (xyz,rpy)

        npoffset = np.abs(np.concatenate([e.sampled_offset.pos.array, e.sampled_offset.orientation.to_euler(encoding='xyz')]))

        for i in range(n_input_vars):
            tmpdict[v_names[i]] = npstd[i]
            tmpdict[v_names_offsets[i]] = npoffset[i]

        tmpdict[o_names[0]] = tf_lookup[e.b_outcome]

        #if e.b_handling_error_likely:
        #    tmpdict[o_names[0]] = 0.5

        tmpdict[o_names[1]] = tf_lookup[e.b_handling_error_likely]

        tmpdict[f_names[0]] = np.sum(npstd[0:3])
        tmpdict[f_names[1]] = np.sum(npstd[3:6])


        tmpdict[f_names[2]] =np.max(npstd[0:3])
        tmpdict[f_names[3]] = np.max(npstd[3:6])

        tmpdict[f_names[4]] = e.sampled_offset.pos.length
        tmpdict[f_names[5]] = e.sampled_offset.orientation.ang_norm

        pdata_list.append(tmpdict)

    pandas_data = pd.DataFrame(pdata_list)

    pandas_data.reset_index(drop=True, inplace=True)

    return pandas_data


def plot_variances(df):

    vi = 0
    oi = 0
    fi = 0

#    for vi in range(6):
        #df.plot(x=v_names[vi], y=o_names[oi], style='o')
#        df.boxplot(column=v_names[vi], by=o_names[oi])

    #for fi in range(2):
        #df.plot(x=f_names[fi], y=o_names[oi], style='o')

    colors = np.where(df[o_names[oi]]==1,'g','k')
    print(f'Succes count: {np.sum(df[o_names[oi]])}')
    #colors = np.where(df[o_names[oi]] == 1, 'y', np.where(df[o_names[1]] == 1,'r','k'))
    # figsave(df.plot.scatter(x=f_names[0], y=f_names[1], c=colors, marker='x'))

    plt_style_scatter(True)
    ax = df.loc[df[o_names[oi]] == 1].plot(x=f_names[0], y=f_names[1], marker='o', c=color(1))
    ax = df.loc[df[o_names[oi]] == 0].plot(x=f_names[0], y=f_names[1], ax=ax, marker='x', c=color(2))
    plt_style_scatter(False)

    figsave(df.plot.scatter(x=f_names[0], y=f_names[1]))

#    figsave(df.plot.scatter(x=f_names[2], y=f_names[3], c=colors, style='x', alpha=0.3))
#    figsave(df.plot.scatter(x=v_names[1], y=v_names[2], c=colors, style='x', alpha=0.3))

#    figsave(df.boxplot(column=f_names[0], by=o_names[oi]))
#    figsave(df.boxplot(column=f_names[1], by=o_names[oi]))

#    for vi in range(6):
        #df.plot(x=v_names[vi], y=o_names[oi], style='o')
#        df.boxplot(column=v_names_offsets[vi], by=o_names[oi])

#    for vi in range(6):
#        df.plot(x=v_names_offsets[vi], y=o_names[oi], style='o')

#    figsave(df.plot.scatter(x=f_names[4],y=f_names[5],c=colors, style='x', alpha=0.3))

#    figsave(df.boxplot(column=f_names[4], by=o_names[oi]))
#    figsave(df.boxplot(column=f_names[5], by=o_names[oi]))


if __name__ == "__main__":
    fn = '/home/klaus/code/pymi2_ws/sim_data/out/testdata_v2_1k.dill'
    figurefolder = '/home/klaus/code/pymi2_ws/sim_data/figures'

    data = load_file(fn)
    relearn(data, bshuffle=False)


    df = data_2_pandas(data)
    #print(df.count())
    #plot_variances(df)




    plt.show(block=True)
