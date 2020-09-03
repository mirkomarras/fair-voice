import os
import shutil
from pydub import AudioSegment

# SOURCE_DIR_PATH is the path of the folder where FairVoice Dataset is stored
SOURCE_DIR_PATH = '/home/hichamlafhouli/FairVoice/'
# DEST_DIR_PATH is the destination path of the converted Dataset
DEST_DIR_PATH = '../FairVoice'


def checkSourceEnv():
    """
    Function used to check the environment
    :return: boolean
    """
    if (os.path.exists(SOURCE_DIR_PATH)):
        print('---directory available---\n')
        return True
    else:
        return False


def createDestEnv():
    """
    Function used to create the destination directory
    :return: NONE
    """
    if (os.path.exists(DEST_DIR_PATH)):
        shutil.rmtree(DEST_DIR_PATH)
        os.mkdir(DEST_DIR_PATH)
    else:
        os.mkdir(DEST_DIR_PATH)


def conv(srcDir, destDir):
    """
    Function used to convert the file inside a specific folder
    :param srcDir: Input directory
    :param destDir: Output directory
    :return: NONE
    """

    # Iterate files inside the input directory
    for file in os.listdir(srcDir):
        # read the audio file
        sound = AudioSegment.from_mp3(os.path.join(srcDir, file))
        # Set the frame rate
        sound.set_frame_rate(16000)
        # Export the audio file converted in .wav with a bitrate of 256bit and 16000Hz of frame rate
        sound.export(os.path.join(destDir, file[:-3]) + 'wav', format="wav", bitrate='256', parameters=['-ar', '16000'])

    print(destDir.split('/')[-2] + '\t ID:  ' + destDir.split('/')[-1] + '\t CONVERTED')


def diveInSource(path):
    """
    Function used to inspect the contents of the foldes
    :param path: path of the input directory
    :return: NONE
    """
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
