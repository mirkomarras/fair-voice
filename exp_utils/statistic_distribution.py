from scipy.stats import ttest_rel
import json
import pandas as pd
import random
import argparse
import numpy as np
from sklearn.metrics import roc_curve

DEST_STATISTIC_RES = '/home/meddameloni/fair-voice/exp/statistics/statistic_distribution_EER.csv'
DEST_STATISTIC_FAR = '/home/meddameloni/fair-voice/exp/statistics/statistic_distribution_FAR.csv'
DEST_STATISTIC_FRR = '/home/meddameloni/fair-voice/exp/statistics/statistic_distribution_FRR.csv'

BEST_RESULTS_PATHS = 'path_best_results.json'

def calculate_threshold(y, y_score):
    """
    Function use to calculate the average threshold
    :param y:               real result
    :param y_score:         predicted result
    :return:                Threshold

    """

    far, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    frr = 1 - tpr
    abs_diffs = np.abs(far - frr)
    min_index = np.argmin(abs_diffs)
    threshold = thresholds[min_index]
    return threshold


def drop_indices(idx_list, data):
    """
    Function used to drop elements in data structure to equalize the number of instances for each sensitive category
    :param idx_list: list of indices to drop
    :param data: main structure where the index is eliminated
    :return: data - the main structure without the dropped indices
    """
    for idx in idx_list:
        data = data.drop(index=idx, axis=0)
    return data


def countFRR(gbu_data, thr):
    """
    Function that calculate the average False Rejection Rate for the list in input grouped by user given a certain
    threshold.
    :param gbu_data: data list grouped by user
    :param thr: value of threshold to consider for check the false negatives
    :return: False Rejection Rate
             List of users presenting cases of false positives
    """
    # Dictionary that contain the amount of false negatives per user
    gbu_res = {}
    # number of cases per user
    gbu_size = gbu_data.size()
    # list used to store FRR per user
    frr_res_users = []
    for name, group in gbu_data:
        for index, row in group.iterrows():
            if (row['simlarity'] < thr):
                if (name in gbu_res.keys()):
                    gbu_res[name] += 1
                else:
                    gbu_res[name] = 1
            else:
                if (name in gbu_res):
                    continue
                else:
                    gbu_res[name] = 0
    for user in gbu_res.keys():
        frr_res_users.append(gbu_res[user]/gbu_size[user])

    return sum(frr_res_users)/len(frr_res_users), frr_res_users


def countFAR(gbu_data, thr):
    """
    Function that calculate the average false acceptance rate for the list in input grouped by user given a certain
    threshold
    :param gbu_data: data list grouped by user
    :param thr: value of threshold to consider for check the false positives
    :return: False Acceptance Rate
             List of users presenting cases of false positives
    """
    # Dictionary that contain the amount of false positives per user
    gbu_res = {}
    # number of cases per user
    gbu_size = gbu_data.size()
    # list used to store FAR per user
    far_res_users = []

    for name, group in gbu_data:
        for index, row in group.iterrows():
            if (row['simlarity'] > thr):
                if (name in gbu_res.keys()):
                    gbu_res[name] += 1
                else:
                    gbu_res[name] = 1
            else:
                if (name in gbu_res):
                    continue
                else:
                    gbu_res[name] = 0
    for user in gbu_res.keys():
        far_res_users.append(gbu_res[user] / gbu_size[user])

    return sum(far_res_users) / len(frr_res_users), far_res_users


def computeResultsFAR(path):
    """
    Main Function used to split result data in order to calculate False Acceptance Rate between sensitive categories
    and to check if there corresponding sensitive groups are similar
    :param path: path of the test results
    :return: far_avg_m -> FAR for male category
             far_avg_f -> FAR for female category
             far_avg_o -> FAR for old category
             far_avg_y -> FAR for young category
             res_mf -> 'Y' if results are similar between Male and Female groups, 'N' otherwise
             res_yo -> 'Y' if results are similar between Old and Young groups, 'N' otherwise
    """
    #Read data from the test's result file
    data = pd.read_csv(path)

    # split column 'audio_1' in two, in order to group by user
    data[['user_1', 'audio_1']] = data['audio_1'].str.rsplit('/', 1, expand=True)

    # Extract the columns, one containing the expected result (label) and one with the predicted results (simlarity)
    label = data.loc[:, 'label']
    similarity = data.loc[:, 'simlarity']

    # Here is taken the average threshold in order to verify later how much false acceptances have occured
    thr = calculate_threshold(label, similarity)

    # Here are taken into account only the labels that are 0, so don't correspond to the right user
    false_labels = data[label == 0].reset_index()

    # Here are divided by sensitive categories
    false_labels_m = false_labels[false_labels.gender_1 == 'male']
    false_labels_f = false_labels[false_labels.gender_1 == 'female']
    false_labels_y = false_labels[false_labels.age_1 == 'young']
    false_labels_o = false_labels[false_labels.age_1 == 'old']

    # Every list is now grouped by specific user (user1)
    tl_gbu_m = false_labels_m.groupby('user_1', axis=0)
    tl_gbu_f = false_labels_f.groupby('user_1', axis=0)
    tl_gbu_y = false_labels_y.groupby('user_1', axis=0)
    tl_gbu_o = false_labels_o.groupby('user_1', axis=0)

    # Elaboration of FAR for the entire cases and list of FAR per user
    far_avg_m, far_avg_m_users = countFAR(tl_gbu_m, thr)
    far_avg_f, far_avg_f_users = countFAR(tl_gbu_f, thr)
    far_avg_y, far_avg_y_users = countFAR(tl_gbu_y, thr)
    far_avg_o, far_avg_o_users = countFAR(tl_gbu_o, thr)

    print('Male Users: {}\tFemale Users: {}'.format(len(far_avg_m_users), len(far_avg_f_users)))
    print('Young Users: {}\tOld Users: {}'.format(len(far_avg_y_users), len(far_avg_o_users)))

    # Here any similarities between sensitive groups are controlled
    res_mf = t_test(far_avg_m_users, far_avg_f_users)
    res_yo = t_test(far_avg_y_users, far_avg_o_users)

    return far_avg_m, far_avg_f, far_avg_y, far_avg_o, res_mf, res_yo


def computeResultsFRR(path):
    """
    Main Function used to split result data in order to calculate False Rejection Rate between sensitive categories
    and to check if there corresponding sensitive groups are similar
    :param path: path of the test results
    :return: frr_avg_m -> FRR for male category
             frr_avg_f -> FRR for female category
             frr_avg_o -> FRR for old category
             frr_avg_y -> FRR for young category
             res_mf -> 'Y' if results are similar between Male and Female groups, 'N' otherwise
             res_yo -> 'Y' if results are similar between Old and Young groups, 'N' otherwise
    """
    #Read data from the test's result file
    data = pd.read_csv(path)

    # split column 'audio_1' in two, in order to group by user
    data[['user_1', 'audio_1']] = data['audio_1'].str.rsplit('/', 1, expand=True)

    # Extract the columns, one containing the expected result (label) and one with the predicted results (simlarity)
    label = data.loc[:, 'label']
    similarity = data.loc[:, 'simlarity']

    # Here is taken the average threshold in order to verify later how much false acceptances have occured
    thr = calculate_threshold(label, similarity)

    # Here are taken into account only the labels that are 0, so don't correspond to the right user
    true_labels = data[label == 1].reset_index()

    # Here are divided by sensitive categories
    true_labels_m = true_labels[true_labels.gender_1 == 'male']
    true_labels_f = true_labels[true_labels.gender_1 == 'female']
    true_labels_y = true_labels[true_labels.age_1 == 'young']
    true_labels_o = true_labels[true_labels.age_1 == 'old']

    # Every list is now grouped by specific user (user1)
    tl_gbu_m = true_labels_m.groupby('user_1', axis=0)
    tl_gbu_f = true_labels_f.groupby('user_1', axis=0)
    tl_gbu_y = true_labels_y.groupby('user_1', axis=0)
    tl_gbu_o = true_labels_o.groupby('user_1', axis=0)

    # Elaboration of FRR for the entire cases and list of FRR per user
    frr_avg_m, frr_avg_m_users = countFRR(tl_gbu_m, thr)
    frr_avg_f, frr_avg_f_users = countFRR(tl_gbu_f, thr)
    frr_avg_y, frr_avg_y_users = countFRR(tl_gbu_y, thr)
    frr_avg_o, frr_avg_o_users = countFRR(tl_gbu_o, thr)

    print('Male Users: {}\tFemale Users: {}'.format(len(frr_avg_m_users), len(frr_avg_f_users)))
    print('Young Users: {}\tOld Users: {}'.format(len(frr_avg_y_users), len(frr_avg_o_users)))

    # Here any similarities between sensitive groups are controlled
    res_mf = t_test(frr_avg_m_users, frr_avg_f_users)
    res_yo = t_test(frr_avg_y_users, frr_avg_o_users)

    return frr_avg_m, frr_avg_f, frr_avg_y, frr_avg_o, res_mf, res_yo


def computeResultsSimilarity(path):
    """
    Function that is used to extract the results divided by sensitive categories. If the corresponding sensitve categories
    (e.g. Male - Female) dont have the same amount of results then a procedure is called to equalize the amount of
    results in order to calculate correctly the statistic distribution between categories.
    :param path: test results path
    :return: the results of the function ttest used to calculate the statistic distribution of the categories.
    """
    data = pd.read_csv(path)

    #split column 'audio_1' in two, in order to group by user
    data[['user_1', 'audio_1']] = data['audio_1'].str.rsplit('/', 1, expand=True)

    #creating views for taking into account similarity score divided by categories
    data_m = data[data.gender_1 == 'male'].reset_index()
    data_f = data[data.gender_1 == 'female'].reset_index()
    data_y = data[data.age_1 == 'young'].reset_index()
    data_o = data[data.age_1 == 'old'].reset_index()

    #calculate the distance between rows in related groups
    dist_mf = len(data_m) - len(data_f)
    dist_yo = len(data_y) - len(data_o)

    # if the length of the related groups are not equals (so we haven't the same amount of results for each corrispondent
    # sensitive category)
    if (dist_mf !=0 or dist_yo != 0):

        print('difference M/F: {}\tdifference Y/O: {}'.format(dist_mf, dist_yo))
        #PROCESS FOR REDUCE INSTANCES FOR MALE AND FEMALE GROUPS
        #make a view for the related categories group by user ID
        gbu_m = data_m.groupby('user_1', axis=0)
        gbu_f = data_f.groupby('user_1', axis=0)

        # if the difference is greater than 0 we're going to drop rows from male set
        # otherwise we're going to drop rows from female set
        choice_group_to_reduce = ''
        list_idx_to_drop = []
        if dist_mf > 0:
            choice_group_to_reduce = 'm'
        elif dist_mf < 0:
            choice_group_to_reduce = 'f'

        #since the distance is different from 0
        while (dist_mf != 0):
            if (dist_mf > 0):
                for gr in gbu_m.indices.keys():
                    if (dist_mf == 0):
                        break
                    check = True
                    while (check):
                        # a random index is choosen from the list to be the one to drop
                        idx_to_drop = random.randint(0, len(gbu_m.indices[gr])-1)
                        if (gbu_m.indices[gr][idx_to_drop] not in list_idx_to_drop):
                            list_idx_to_drop.append(gbu_m.indices[gr][idx_to_drop])
                            check = False
                    dist_mf -= 1
            elif (dist_mf < 0):
                for gr in gbu_f.indices.keys():
                    if (dist_mf == 0):
                        break
                    check = True
                    while (check):
                        idx_to_drop = random.randint(0, len(gbu_f.indices[gr]) - 1)
                        if (gbu_f.indices[gr][idx_to_drop] not in list_idx_to_drop):
                            list_idx_to_drop.append(gbu_f.indices[gr][idx_to_drop])
                            check = False
                    dist_mf += 1
            else:
                break
        # after the list with the indices to drop is being populated the indices are dropped based on the exceeded
        # sensitive category
        if choice_group_to_reduce == 'm':
            data_m = drop_indices(list_idx_to_drop, data_m)
            print('> Male set reduced at {} rows'.format(len(data_m)))
        elif choice_group_to_reduce == 'f':
            data_f = drop_indices(list_idx_to_drop, data_f)
            print('> Female set reduced at {} rows'.format(len(data_f)))

        #same procedure seen above but for young and old categories
        gbu_y = data_y.groupby('user_1', axis=0)
        gbu_o = data_o.groupby('user_1', axis=0)

        choice_group_to_reduce = ''
        list_idx_to_drop = []
        if dist_yo > 0:
            choice_group_to_reduce = 'y'
        elif dist_yo < 0:
            choice_group_to_reduce = 'o'

        while (dist_yo != 0):
            if (dist_yo > 0):
                for gr in gbu_y.indices.keys():
                    if (dist_yo == 0):
                        break
                    check = True
                    while (check):
                        idx_to_drop = random.randint(0, len(gbu_y.indices[gr]) - 1)
                        if (gbu_y.indices[gr][idx_to_drop] not in list_idx_to_drop):
                            list_idx_to_drop.append(gbu_y.indices[gr][idx_to_drop])
                            check = False
                    dist_yo -= 1
            elif (dist_yo < 0):
                for gr in gbu_o.indices.keys():
                    if (dist_yo == 0):
                        break
                    check = True
                    while (check):
                        idx_to_drop = random.randint(0, len(gbu_o.indices[gr]) - 1)
                        if (gbu_o.indices[gr][idx_to_drop] not in list_idx_to_drop):
                            list_idx_to_drop.append(gbu_o.indices[gr][idx_to_drop])
                            check = False
                    dist_yo += 1
            else:
                break

        if choice_group_to_reduce == 'y':
            data_y = drop_indices(list_idx_to_drop, data_y)
            print('> Young set reduced at {} rows'.format(len(data_y)))
        elif choice_group_to_reduce == 'o':
            data_o = drop_indices(list_idx_to_drop, data_o)
            print('> Old set reduced at {} rows'.format(len(data_o)))

    similarity_m = data_m.loc[:, 'simlarity']
    similarity_f = data_f.loc[:, 'simlarity']
    similarity_y = data_y.loc[:, 'simlarity']
    similarity_o = data_o.loc[:, 'simlarity']

    print('MALE: {} | FEMALE: {}\tYOUNG: {} | OLD: {}'.format(len(similarity_m), len(similarity_f), len(similarity_y), len(similarity_o)))

    return t_test(similarity_m, similarity_f), t_test(similarity_y, similarity_o)


def t_test(data1, data2, return_verbose=True):
    """
    This function is used to do the t_test, basically is used to check if the lists in input have the same statistical
    distribution, so how much similar are the two groups considered and if the differences between each other could have
    happened by chance.
    Low score tells us that the group is similar. The average score is 0.05.
    :param data1: first list to compare
    :param data2: second list to compare
    :param return_verbose: NOT CONSIDERED
    :return: a string 'Y' if the lists are similar otherwise a string 'N'.
    """
    stat, p = ttest_rel(data1, data2)
    stats_str = 'stat={0:.3f}, p={1:.3f}'.format(stat, p)
    print(stats_str)
    if p > 0.05:
        return 'Y'
    else:
        return 'N'

    # if return_verbose:
    #     return stats_str, p_str


def main():
    parser = argparse.ArgumentParser(description='Operations utils for test results elaboration')

    # Is possible to choose which modality to use (EER, FRR, FAR)
    parser.add_argument('--eer', dest='eer_mode', default=True, type=bool,
                        action='store', help='Base path for results')
    parser.add_argument('--far', dest='far_mode', default=False, type=bool,
                        action='store', help='Base path for destination folder results')
    parser.add_argument('--frr', dest='frr_mode', default=False, type=bool,
                        action='store', help='Base path for destination folder results')

    args = parser.parse_args()

    if args.far_mode == True or args.frr_mode == True:
        args.eer_mode = False

    if (args.eer_mode):
        # EER MODE ON for check stastic distribution over this measure
        # final .csv result columns structure is setted
        columns = ['test file', 'network', 'train file', 'accuracy', 'distribution M/F', 'distribution Y/O']
        recordsToLoad = []
        # Here are imported all the paths of .csv results considered from an external .json file
        with open(BEST_RESULTS_PATHS) as json_file:
            br = json.load(json_file)
            # iterate paths loaded from the .json file
            for tests in br.keys():
                print('>  Elaborating results for {}'.format(tests))
                for test_path in br[tests]:
                    res_test_namefile = test_path.split('/')[-1]
                    print('>  Compute distributions for {}'.format(res_test_namefile.split('_')[1]))

                    res_mf, res_yo = computeResultsSimilarity(test_path)
                    recordToAdd = [
                        tests,
                        res_test_namefile.split('_')[0],
                        res_test_namefile.split('_')[1],
                        res_test_namefile.split('_')[2],
                        res_mf,
                        res_yo
                    ]
                    recordsToLoad.append(recordToAdd)

        df = pd.DataFrame(recordsToLoad)
        df.columns = columns
        df.to_csv(DEST_STATISTIC_RES, index=False)

    elif (args.far_mode):
        print('FAR MODE ON')
        columns = ['test file',
                   'network',
                   'train file',
                   'accuracy',
                   'count M FAR',
                   'count F FAR',
                   'count Y FAR',
                   'count O FAR',
                   'Stats correl M/F',
                   'Stats correl Y/O']
        recordsToLoad = []
        with open('path_best_results.json') as json_file:
            br = json.load(json_file)

            for tests in br.keys():
                print('>  Elaborating results for {}'.format(tests))
                for test_path in br[tests]:
                    res_test_namefile = test_path.split('/')[-1]
                    print('>  Compute distributions for {}'.format(res_test_namefile.split('_')[1]))
                    count_m, count_f, count_y, count_o, res_stat_mf, res_stat_yo = computeResultsFAR(test_path)
                    recordToAdd = [
                        tests,
                        res_test_namefile.split('_')[0],
                        res_test_namefile.split('_')[1],
                        res_test_namefile.split('_')[2],
                        count_m,
                        count_f,
                        count_y,
                        count_o,
                        res_stat_mf,
                        res_stat_yo
                    ]
                    recordsToLoad.append(recordToAdd)
        df = pd.DataFrame(recordsToLoad)
        df.columns = columns
        df.to_csv(DEST_STATISTIC_FAR, index=False)

    elif args.frr_mode:
        print('FRR MODE ON')
        columns = ['test file',
                   'network',
                   'train file',
                   'accuracy',
                   'count M FRR',
                   'count F FRR',
                   'count Y FRR',
                   'count O FRR',
                   'Stats correl M/F FRR',
                   'Stats correl Y/O FRR']
        recordsToLoad = []
        with open('path_best_results.json') as json_file:
            br = json.load(json_file)

            for tests in br.keys():
                print('>  Elaborating results for {}'.format(tests))
                for test_path in br[tests]:
                    res_test_namefile = test_path.split('/')[-1]
                    print('>  Compute distributions for {}'.format(res_test_namefile.split('_')[1]))
                    count_m, count_f, count_y, count_o, res_stat_mf, res_stat_yo = computeResultsFRR(test_path)
                    recordToAdd = [
                        tests,
                        res_test_namefile.split('_')[0],
                        res_test_namefile.split('_')[1],
                        res_test_namefile.split('_')[2],
                        count_m,
                        count_f,
                        count_y,
                        count_o,
                        res_stat_mf,
                        res_stat_yo
                    ]
                    recordsToLoad.append(recordToAdd)
        df = pd.DataFrame(recordsToLoad)
        df.columns = columns
        df.to_csv(DEST_STATISTIC_FRR, index=False)

if __name__ == "__main__":
   main()
