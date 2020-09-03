import os
import pandas as pd


VOX_CELEB_TEST_FILE_PATH = '/home/meddameloni/test_voxceleb_pairs.txt'
BASE_TEST_FOLDER = '/home/meddameloni/fair-voice/exp/test/'

def create_csv_file(aud1, aud2, label):
    print('\n\n> creating csv for test...\n')
    csv_struct = {
        'audio_1': [],
        'audio_2': [],
        'age_1': [],
        'age_2': [],
        'gender_1': [],
        'gender_2': [],
        'label': []
    }

    df = pd.DataFrame(csv_struct)

    df['audio_1'] = aud1
    df['audio_2'] = aud2
    df['age_1'] = ''
    df['age_2'] = ''
    df['gender_1'] = ''
    df['gender_2'] = ''
    df['label'] = label

    csv_path = os.path.join(BASE_TEST_FOLDER, 'Voxceleb-test.csv')
    df.to_csv(csv_path, index=False)

def main():
    audio_1, audio_2, label = [], [], []
    with open(VOX_CELEB_TEST_FILE_PATH) as fp:
        print('> Loading pairs...')
        pairs_count = 0
        for line in fp:
            label.append(line.split(' ')[0].rstrip())
            audio_1.append(line.split(' ')[1].rstrip())
            audio_2.append(line.split(' ')[2].rstrip())
            pairs_count += 1
        print('\n\n> Pairs LOADED!')

        create_csv_file (audio_1, audio_2, label)
        print('> TEST FILE generated in:\t {}\n> TOTAL PAIRS:\t\t\t {}'.format(BASE_TEST_FOLDER, pairs_count))

if __name__ == '__main__':
    main()
