import datetime
import os
import pandas as pd
import random
import argparse

#TODO List

def findMinimumNumPerCategories(df_female,df_male):

    numb_for_age_ranges = {}
    total=0
    for ca in df_female.age.unique():

        n_w=df_female[df_female==ca].age.count();
        n_m=df_male[df_male==ca].age.count();
        if n_w <= n_m:
            if n_w >= 10  :
                numb_for_age_ranges[ca]=n_w;
                total+=n_w
            else:
                print('This category is been removed:',ca);
        else:
            if n_m >= 10:
                numb_for_age_ranges[ca]=n_m;
                total+=n_m
            else:
                print('This category is been removed:',ca);
    print('----------------------------------------')
    print('Minimum number that can be selected for category')
    print("Number of user for category for gender without rebalancing: ",total);
    print("minimo numero per categoria:",numb_for_age_ranges);
    print('----------------------------------------')
    return numb_for_age_ranges

o
def balanceNumberCategories(numb_for_age_ranges,num):

    tmp=numb_for_age_ranges.copy();
    res={}
    while(num > 0):
        for i in tmp.keys():
            if tmp[i] > 0:
                tmp[i]-=1;
                num-=1;
            if num == 0:
                break;

    for ca in numb_for_age_ranges.keys():
        res[ca]=numb_for_age_ranges[ca]-tmp[ca];


    return res;



def splitDataset(d_ages,df_female,df_male,num_usr_catg=4,min_num_rec=14):

    df_female=df_female.sample(frac=1).reset_index(drop=True);
    df_male=df_male.sample(frac=1).reset_index(drop=True);



    df_test=pd.DataFrame(columns=df_female.columns);
    df_male_f=pd.DataFrame(columns=df_female.columns);
    df_female_f=pd.DataFrame(columns=df_female.columns);


    for ca in d_ages.keys():

        test_fm=df_female[df_female.age == ca];
        test_fm_sc=test_fm[test_fm.n_sample >= min_num_rec];
        test_fm_scp=test_fm_sc.sample(frac=1).reset_index(drop=True);


        test_ml=df_male[df_male.age == ca];
        test_ml_sc=test_ml[test_ml.n_sample >= min_num_rec];
        test_ml_scp=test_ml_sc.sample(frac=1).reset_index(drop=True);

        if len(test_fm_scp) < num_usr_catg or len(test_fm_scp) < num_usr_catg:
            raise Exception('Error file of the test cannot be created because \
            there is not there are not enough speaker with at least {} sample'.format(num_usr_catg))
        test_fm_f=test_fm_scp.loc[0:num_usr_catg-1,:].copy();
        test_ml_f=test_ml_scp.loc[0:num_usr_catg-1,:].copy();

        df_test=pd.concat([df_test,test_fm_f],ignore_index=True)
        df_test=pd.concat([df_test,test_ml_f],ignore_index=True)


    for usr in df_test.id_user:
        df_female.drop(df_female.loc[df_female['id_user']== usr ].index, inplace=True)
        df_male.drop(df_male.loc[df_male['id_user']== usr ].index, inplace=True)


    for ca in d_ages.keys():
        train_fm_sc=df_female[df_female.age == ca].reindex();
        train_fm_scp=train_fm_sc.sample(frac=1).reset_index(drop=True)
        train_fm_f=train_fm_scp.loc[0:d_ages[ca]-(num_usr_catg+1),:].copy();
        df_female_f=pd.concat([df_female_f,train_fm_f],ignore_index=True)

        train_ml_sc=df_male[df_male.age == ca].reindex();
        train_ml_scp=train_ml_sc.sample(frac=1).reset_index(drop=True)
        train_ml_f=train_ml_scp.loc[0:d_ages[ca]-(num_usr_catg+1),:].copy();
        df_male_f=pd.concat([df_male_f,train_ml_f],ignore_index=True)



    df_train=pd.concat([df_female_f,df_male_f],ignore_index=True)



    print('----------------------------------------')
    print('Division')
    print("train size:",df_train.shape)
    print("----------------")
    print("test size:",df_test.shape)
    print('----------------------------------------')

    return df_train,df_test

def computeTrain(train):

    train['label']=range(0,len(train),1)

    tmp_f=train[train.gender =='female'];
    num_f=tmp_f.n_sample.sum();
    print("Female sample : ",num_f)

    tmp_m=train[train.gender =='male'];
    num_m=tmp_m.n_sample.sum();
    print("Male sample : ",num_m)

    train_info=pd.DataFrame();
    dictionaire={}
    dict_label={}
    dict_age={}
    dict_gender={}

    if num_m > num_f :
        for idx in tmp_m.index:
            dictionaire[tmp_m.id_user[idx]]=tmp_m.n_sample[idx];
            dict_label[tmp_m.id_user[idx]]=tmp_m.label[idx];
            dict_age[tmp_m.id_user[idx]]=tmp_m.age[idx];
            dict_gender[tmp_m.id_user[idx]]=tmp_m.gender[idx];
        dictionaire=balanceNumberCategories(dictionaire,num_f);
        df_temp=tmp_f;
    else:
        for idx in tmp_f.index:
            dictionaire[tmp_f.id_user[idx]]=tmp_f.n_sample[idx];
            dict_label[tmp_f.id_user[idx]]=tmp_f.label[idx];
            dict_age[tmp_f.id_user[idx]]=tmp_f.age[idx];
            dict_gender[tmp_f.id_user[idx]]=tmp_f.gender[idx]
        dictionaire=balanceNumberCategories(dictionaire,num_m);
        df_temp=tmp_m;

    train.loc[:,['id_user','label','n_sample','gender','age']].reindex()
    pair_tr ={}
    pair_tr_info={}

    index=0;
    for idx in df_temp.index:
        for smp in range(0,df_temp.loc[idx,'n_sample'],1):
            audio=df_temp.loc[idx,'id_user']+'/'+"audio"+f'{smp:05}'+".mp3"
            pair_tr[index]={'audio':audio,'label':df_temp.loc[idx,'label']}
            pair_tr_info[index]={'audio':audio,'user':df_temp.loc[idx,'id_user'],\
            'gender':df_temp.loc[idx,'gender'],'age':df_temp.loc[idx,'age']}
            index+=1;


    for idx in dictionaire.keys():
        for smp in range(0,dictionaire[idx],1):
            audio=idx+'/'+"audio"+f'{smp:05}'+".mp3"
            pair_tr[index]={'audio':audio,'label':dict_label[idx]}
            pair_tr_info[index]={'audio':audio,'user':idx,\
            'gender':dict_gender[idx],'age':dict_age[idx]}
            index+=1;


    train = pd.DataFrame.from_dict(pair_tr, "index")
    train_info = pd.DataFrame.from_dict(pair_tr_info, "index")
    train_info.to_csv("check.csv",index=False)

    return train

def combinationForPositiveTuple(num_audio,n=10,k=4):
    available_tuples=[None]*num_audio;
    for i in range(0,num_audio,1):
        tuple=[]
        for j in range(i+1,num_audio,1):
            tuple.append([i,j])
        available_tuples[i]=tuple
    final_combinations=[None]*n
    for i in range(n):
        random.shuffle(available_tuples[i])
        tuple_for_spk=[]
        for _ in range(0,k,1):
            tp=available_tuples[i].pop();
            tuple_for_spk.append(tp)
        for _ in range(0,len(available_tuples[i]),1):
            v=available_tuples[i].pop();
            available_tuples[v[1]].append([v[1],v[0]])
        final_combinations[i]=tuple_for_spk
    return final_combinations

def addRowToTest(speaker_1,speakers_2,triple_ts,idx,index,i):
    rnd_usr=random.randint(0,speakers_2.id_user.count()-1);
    audio_1=speaker_1.loc[idx,'id_user']+'/'+"audio"+f'{i:05}'+".mp3"
    rnd_smp=random.randint(0,speakers_2.loc[rnd_usr,'n_sample']-1)
    audio_2=speakers_2.loc[rnd_usr,'id_user']+'/'+"audio"+f'{rnd_smp:05}'+".mp3"
    triple_ts[index]={'audio_1':audio_1,'audio_2':audio_2,'age_1':speaker_1.loc[idx,'age'],\
                      'age_2':speakers_2.loc[rnd_usr,'age'],'gender_1':speaker_1.loc[idx,'gender'],\
                      'gender_2':speakers_2.loc[rnd_usr,'gender'],'label':0}
    index+=1
    return triple_ts, index


def computeTest(test):
    index=0;

    print("Number of user in test file",test.id_user.count())

    index=0;
    triple_ts={}

    for idx in test.index:
        num_audio=test.loc[idx,'n_sample']-1;
        combination_list=combinationForPositiveTuple(num_audio);
        for lis in combination_list:
            for a in lis:
                audio_1=test.loc[idx,'id_user']+'/'+"audio"+f'{a[0]:05}'+".mp3"
                audio_2=test.loc[idx,'id_user']+'/'+"audio"+f'{a[1]:05}'+".mp3"
                triple_ts[index]={'audio_1':audio_1,'audio_2':audio_2,'age_1':test.loc[idx,'age'],\
                               'age_2':"",'gender_1':test.loc[idx,'gender'],'gender_2':"",'label':1}
                index+=1


        test_without_user=test.copy();
        test_without_user.drop(test_without_user.loc[test_without_user['id_user']==test.loc[idx,'id_user']].index, inplace=True)
        test_wu=test_without_user.sample(frac=1).reset_index(drop=True);

        test_wu_f=test_wu[test_wu.gender=='female']
        test_wu_m=test_wu[test_wu.gender=='male']
        test_wu_f_sr=test_wu_f[ test_wu_f.age == test.age[idx] ].reset_index()
        test_wu_m_sr=test_wu_m[ test_wu_m.age == test.age[idx] ].reset_index()
        test_wu_f_dr=test_wu_f[ test_wu_f.age != test.age[idx] ].reset_index()
        test_wu_m_dr=test_wu_m[ test_wu_m.age != test.age[idx] ].reset_index()

        for i in range(0,10,1):
            triple_ts, index=addRowToTest(test,test_wu_f_sr,triple_ts,idx,index,i)
            triple_ts, index=addRowToTest(test,test_wu_m_sr,triple_ts,idx,index,i)
            triple_ts, index=addRowToTest(test,test_wu_f_dr,triple_ts,idx,index,i)
            triple_ts, index=addRowToTest(test,test_wu_m_dr,triple_ts,idx,index,i)



    test = pd.DataFrame.from_dict(triple_ts, "index")

    return test;
def split_function(path="",num_of_spk=100,language="",num_sample=5):


    try:
        data=pd.read_csv(path)
    except IOError:
        print("Error: can\'t find file or read data")
    else:
        if len(data[data.language_l1==language]) == 0:
                print("Error: Try to use a upper-case for the first letter")
        else:
            print('Ok: Passed all checks')
            date = datetime.datetime.now()
            path=str(date.day)+'_'+str(date.month)+'_'+str(date.year)+\
            '_'+str(date.hour)+'_'+str(date.minute)+'_'+language
            os.mkdir(path);
            data['label']=range(0,len(data),1)
            data['age']=data.age.map({'fifties': 'old', 'fourties': 'old', \
                'thirties': 'young', 'sixties': 'old', 'twenties':'young',\
                'teens': 'young', 'seventies': 'old'})
            temp=data[data.language_l1==language];
            temp=temp[temp.n_sample >= num_sample ]
            temp_fm=temp[temp.gender=='female']
            temp_ml=temp[temp.gender=='male']
            print("Number of female user:",temp_fm.id_user.count())
            print("Number of male user:",temp_ml.id_user.count())
            print('----------------------------------------')
            print("Information about the age of female user:\n",temp_fm.groupby('age')['id_user'].nunique())
            print("Information about the age of male user:\n",temp_ml.groupby('age')['id_user'].nunique())


            numb_for_age_ranges=findMinimumNumPerCategories(temp_fm,temp_ml);
            print(numb_for_age_ranges)
            num_max_spk=0;
            for ca in numb_for_age_ranges.keys():
                num_max_spk+=numb_for_age_ranges[ca];
            print('Max numbero of user that can be used:',(num_max_spk*2)-40);

            if num_of_spk <= num_max_spk:
                numb_for_age_ranges_bl=balanceNumberCategories(numb_for_age_ranges,num_of_spk)
                print('----------------------------------------')
                print('Balanced Combination')
                print("results",numb_for_age_ranges_bl)
                train,test=splitDataset(numb_for_age_ranges_bl,temp_fm,temp_ml)
                train.to_csv(path+'/train_info.csv');
                test.to_csv(path+'/test_info.csv');
                if(len(numb_for_age_ranges_bl)> 1):
                    train_result=computeTrain(train)
                    test_result=computeTest(test)
                    print('Excel file created for',language)
                    print(path+'/'+language+"_train.csv")
                    train_result.to_csv(path+'/'+language+"_train.csv",index=False)
                    test_result.to_csv(path+'/'+language+"_test.csv",index=False)
                else:
                    print('Error: There are not enough user ')
            else:
                print('Error: The number of user selected must be less or equal ',num_max_spk)

def main():

    parser = argparse.ArgumentParser(description='Generate file Train and test')
    parser.add_argument('--lan', dest='lan', default='English',choices=['English','Spanish','French','German'], type=str, action='store', help='')
    parser.add_argument('--file_path', dest='file_path', default="C:/Users/M1/Documents/FairVoice/metadata/metadata.csv", type=str, action='store', help='Base path for validation trials')
    parser.add_argument('--num_of_spk', dest='num_of_spk', default=1020, type=int, action='store', help='')
    parser.add_argument('--min_samples', dest='min_samples', default=5, type=int, action='store', help='')
    args = parser.parse_args()
    split_function(path=args.file_path,language=args.lan,num_of_spk=args.num_of_spk,num_sample=args.min_samples)

if __name__== "__main__":
  main()
