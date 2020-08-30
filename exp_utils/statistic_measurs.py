import scipy.stats
import itertools
import os
import pandas as pd

out_path = '/home/meddameloni/dl-fair-voice/exp/statistics'


def sample_info_statistics(path):
    df = pd.read_csv(path, sep=',', encoding='utf-8')
    df_group = df.groupby('group')
    gr_comb = itertools.combinations(df_group.groups, 2)
    gr_comb = [comb for comb in gr_comb]
    df_stats_cols = list(df.loc[:, "n_sample":"noise_avg"].columns)

    out_cols = ["stats_value", "p_result"]
    multi_index_len = len(df_stats_cols)
    out_multi_index = pd.MultiIndex.from_arrays([
        list(itertools.chain.from_iterable(map(lambda comb: [comb]*multi_index_len,
                                               gr_comb)
                                           )),
        df_stats_cols * len(gr_comb)
    ])

    data = []
    for grs in gr_comb:
        gr1 = df_group.get_group(grs[0])
        gr2 = df_group.get_group(grs[1])

        for col in df_stats_cols:
            stats_str, p_str = t_test(gr1[col].to_numpy(), gr2[col].to_numpy(), return_verbose=True)
            data.append([stats_str, p_str])

    out_df = pd.DataFrame(data, columns=out_cols, index=out_multi_index)

    with open(os.path.join(out_path, os.path.splitext(os.path.basename(path))[0] + '_statistics.txt'), 'w') as out_file:
        out_file.write(out_df.to_string())


def t_test(data1, data2, return_verbose=True):
    stat, p = scipy.stats.ttest_rel(data1, data2)
    stats_str = 'stat={0:.3f}, p={1:.3f}'.format(stat, p)
    print(stats_str)
    if p > 0.05:
        p_str = 'Probably the same distribution'
        print(p_str)
    else:
        p_str = 'Probably different distributions'
        print(p_str)

    if return_verbose:
        return stats_str, p_str


if __name__ == "__main__":
    for file in ["English-test1", "English-test2", "English-test3", "Spanish-test1", "Spanish-test2", "Spanish-test3"]:
        sample_info_statistics('/home/meddameloni/FairVoice/metadata/sample_info/sample_info_{}.csv'.format(file))
