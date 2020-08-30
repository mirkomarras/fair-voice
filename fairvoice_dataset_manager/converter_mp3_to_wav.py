import os
import shutil
from pydub import AudioSegment


SOURCE_DIR_PATH = '/home/hichamlafhouli/FairVoice/'
DEST_DIR_PATH = '../FairVoice'


def checkSourceEnv():
    if (os.path.exists(SOURCE_DIR_PATH)):
        print('---directory available---\n')
        return True
    else:
        return False


def createDestEnv():
    if (os.path.exists(DEST_DIR_PATH)):
        shutil.rmtree(DEST_DIR_PATH)
        os.mkdir(DEST_DIR_PATH)
    else:
        os.mkdir(DEST_DIR_PATH)


def conv(srcDir, destDir):
    for file in os.listdir(srcDir):
        sound = AudioSegment.from_mp3(os.path.join(srcDir, file))
        sound.set_frame_rate(16000)
        sound.export(os.path.join(destDir, file[:-3]) + 'wav', format="wav", bitrate='256', parameters=['-ar', '16000'])
    print(destDir.split('/')[-2] + '\t ID:  ' + destDir.split('/')[-1] + '\t CONVERTED')


def diveInSource(path):
    print('List elements in Origianl Dataset:')
    for dir in os.listdir(path):
        if(os.path.isdir(os.path.join(path,dir))):
            if (os.path.exists(DEST_DIR_PATH)):
                os.mkdir(os.path.join(DEST_DIR_PATH, dir))
                with os.scandir(os.path.join(path,dir)) as languageFolder:
                    for l_dir in languageFolder:
                        if (os.path.isdir(l_dir)):
                            destIDfolderPath = os.path.join(os.path.join(DEST_DIR_PATH, dir), l_dir.name)
                            os.mkdir(destIDfolderPath)
                            conv(l_dir.path, destIDfolderPath)
        print('>  ' + dir + '\t created references in destination folder')


def mp3toWavConverter ():
    print('\n> Check source directory:')
    assert checkSourceEnv(), "something wrong with source directory"
    createDestEnv()
    print('> Destination directory ready!')
    print('\n\n> Start Conversion:')
    diveInSource(SOURCE_DIR_PATH)
    print('\n> CONVERSION COMPLETE!')


if __name__ == '__main__':
    mp3toWavConverter()
