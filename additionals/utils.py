import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt



#https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
def calculate_eer_v2(y, y_score):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def calculate_eer(y,y_score):
    far, tpr, thresholds =roc_curve(y, y_score, pos_label=1)
    frr= 1- tpr
    abs_diffs = np.abs(far - frr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((far[min_index], frr[min_index]))
    thresh=thresholds[min_index]
    return min_index,far,frr,far[min_index],frr[min_index],eer, thresh,thresholds


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def draw_err(thr,frr,far,x_e,y_e,title):
    fig, ax = plt.subplots(1,1);
    fig = plt.figure();
    ax.plot(thr[1:],frr[1:],label='Frr');
    ax.plot(thr[1:],far[1:],label='Far');
    ax.set_xlabel('Thresholds',fontsize=16);
    ax.set_ylabel('FAR/FRR',fontsize=16);
    ax.plot(x_e,y_e,'rs');
    ax.text(x_e-0.15, y_e, "Threshold %0.2f" %round((x_e),2), ha="center",color='r');
    ax.text(x_e+0.1, y_e, "Eer %0.2f" %round((y_e*100),2), ha="center",color='r',);
    ax.set_title(title,fontsize=18);
    ax.legend();

def draw_err_age(thrs,thrms,frr_m,far_m,frr_f,far_f,thr,eer,thrm,eerm,title):
    fig, ax = plt.subplots(1,1);
    ax.plot(thrs[1:],frr_m[1:],label='FRR OLD');
    ax.plot(thrs[1:],far_m[1:],label='FAR OLD');
    ax.plot(thrms[1:],frr_f[1:],label='FRR YOUNG');
    ax.plot(thrms[1:],far_f[1:],label='FAR YOUNG');
    ax.set_ylabel('FAR/FRR',fontsize=16);
    ax.set_xlabel('Thresholds',fontsize=16);
    ax.text(thrm-0.02, 0.8, "%0.2f EER Young" %round((eerm*100),2), ha="center",color='#33c221',rotation=90);
    ax.plot(thrm,eerm,'s',color='#33c221');
    ax.plot(thr,eer,'s',color='#808080');
    ax.axvline(x=thrm,color='#33c221',linestyle='--')
    ax.axvline(x=thr,color='#808080',ls='--')
    ax.text(thr-0.02, 0.8, "%0.2f EER Old" %round((eer*100),2), ha="center",color='#808080',rotation=90);
    #ax.set_title(title,fontsize=18);
    ax.legend();

def draw_err_gender(thrs,thrms,frr_m,far_m,frr_f,far_f,thr,eer,thrm,eerm,title):
    fig, ax = plt.subplots(1,1);
    ax.plot(thrs[1:],frr_m[1:],label='FRR MALE');
    ax.plot(thrs[1:],far_m[1:],label='FAR MALE');
    ax.plot(thrms[1:],frr_f[1:],label='FRR FEMALE');
    ax.plot(thrms[1:],far_f[1:],label='FAR FEMALE');
    ax.set_xlabel('Thresholds',fontsize=16);
    ax.set_ylabel('FAR/FRR',fontsize=16);
    ax.text(thrm-0.02, 0.8, "%0.2f EER Female" %round((eerm*100),2), ha="center",color='#ffa6c1',rotation=90);
    ax.plot(thrm,eerm,'s',color='#ffa6c1');
    ax.plot(thr,eer,'s',color='#0090cc');
    ax.axvline(x=thrm,color='#ffb6c1',linestyle='--')
    ax.axvline(x=thr,color='#0090cc',ls='--')
    ax.text(thr-0.02, 0.8, "%0.2f EER Male" %round((eer*100),2), ha="center",color='#0090cc',rotation=90);
    #ax.set_title(title,fontsize=18);
    ax.legend();

def utils_graph(data,gender,start,cl):
    from collections import Counter
    d_m=data[data.gender_1==gender]
    d_m=d_m[d_m.label==cl]
    predict=d_m['label'] == d_m['classe_s']
    labels, values = zip(*Counter(predict).items())
    labs=[]
    for l in labels:
        if l:
            labs.append(start+'-right');
        else:
            labs.append(start+'-wrong')
    return labs,values

def prediction_on_actual_outcome(data,lab):
    df_ml_bpc=data[data.gender_1=='male']
    df_ml_bpc=df_ml_bpc[df_ml_bpc.label==lab]
    predict_ml=df_ml_bpc['label'] != df_ml_bpc['classe_s']
    #print(sum(predict_ml))
    df_fm_bpc=data[data.gender_1=='female']
    df_fm_bpc=df_fm_bpc[df_fm_bpc.label==lab]
    predict_fm=df_fm_bpc['label'] != df_fm_bpc['classe_s']
    #print(sum(predict_fm))
    return predict_ml, predict_fm;

def utils_scatter(data):
    list_id=[]
    for aud in data.audio_1:
        list_id.append(aud[0:7])
    list_set = set(list_id)
    list_id = (list(list_set))

    mean_pos=[None]*len(list_id)
    mean_neg=[None]*len(list_id)
    index=0;
    for ids in list_id:
        tmp_s=data[data['audio_1'].str.contains(ids)]
        tmp_s_p=tmp_s[ tmp_s['label'] == 1 ];
        tmp_s_n=tmp_s[ tmp_s['label'] == 0 ];
        mean_pos[index]=tmp_s_p['simlarity'].mean();
        mean_neg[index]=tmp_s_n['simlarity'].mean();
        index+=1;
    return mean_pos, mean_neg

def play_music(music_file, volume=0.8):
    import pygame as pg
    '''
    stream music with mixer.music module in a blocking manner
    this will stream the sound from disk while playing
    '''
    # set up the mixer
    freq = 44100     # audio CD quality
    bitsize = -16    # unsigned 16 bit
    channels = 2     # 1 is mono, 2 is stereo
    buffer = 2048    # number of samples (experiment to get best sound)
    pg.mixer.init(freq, bitsize, channels, buffer)
    # volume value 0.0 to 1.0
    pg.mixer.music.set_volume(volume)
    clock = pg.time.Clock()
    try:
        pg.mixer.music.load(music_file)
        print("Music file {} loaded!".format(music_file))
    except pg.error:
        print("File {} not found! ({})".format(music_file, pg.get_error()))
        return
    pg.mixer.music.play()
    while pg.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)
