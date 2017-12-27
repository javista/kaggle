# 중요한건 dataset에 yes, no, up, down, left, right, on, off, stop, go 만 일단 있도록.
# 추가적으로 Parsing할 때 noise를 추가토록 한다.


import librosa # 푸리에 트랜스폼을 위한 패키지
import numpy as np
import scipy.signal
import torch
import torchaudio # wav 파일을 load 하기 위한 패키지
import copy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
import os

def load_audio(path):
    """
    Input :
        wav 파일 path. 자료형 : str
    Output :
        오디오 파일의 numpy 형태. 자료형 : np
    """
    sound, _ = torchaudio.load(path)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # if multiple channels, average
    return sound

class SpectogramParser(object):
    """
    Fourier transformation parser.
    spectogramparser는 spectogramDataset에 상속되며, 모든 wav 파일을 spectogram으로 파싱하는 클래스이다.

    Param :
        audio_conf - 오디오 특성의 딕셔너리. spectogram화를 위한 window 방식, window size, stride, 음성의 rate 가 들어있어야 한다. 자료형 : dict
            예)
            audio_conf = {}
            audio_conf["sample_rate"] = torchaudio.load("./sample.wav")[1] # torchaudio의 2번째 return value는 해당 wav 파일의 rate이다.
            audio_conf["window_size"] = 0.02
            audio_conf["window_stride"] = 0.01
            audio_conf["window"] = scipy.signal.hamming
    """

    def __init__(self, audio_conf):
        super(SpectogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = audio_conf['window']


    def parse_audio(self, audio_path):
        """
        parsing 함수.
        Input :
            audio_path. 자료형 : str or list
        output :
            해당 path의 spectogram, 자료형 : FloatTensor (MHz + 1, len)
        """
        if(type(audio_path)==str):
            y = load_audio(audio_path)
            n_fft = int(self.sample_rate * self.window_size)
            win_length = n_fft
            hop_length = int(self.sample_rate * self.window_stride)
            # STFT
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=self.window)
            spect, phase = librosa.magphase(D)
            # S = log(S+1)
            spect = np.log1p(spect)
            spect = torch.FloatTensor(spect)
            return spect
        else:
            print("wrong type of audiopath")
            return -1

class SpectogramDataset(Dataset, SpectogramParser):
    """
    pytorch의 dataset 구현. torch의 Dataset, SpectogramParser를 상속받는다.
    또한 *parse_transcript* 함수를 보면, 해당 text 가 index로 리턴이 되어야 한다. 이 또한 Data loader에 구현이 되는 것이다.

    Input :
        audio_conf - 부모 클래스 spectogramparser 초기화를 위한 인자. 자료형 : dict
        manifest_filepath - ./data/train/audio/ 처럼 실제 마지막 나눠져있는 폴더들이 들어가 있는 곳
    """

    def __init__(self, audio_conf, manifest_filepath,train=True):
        """
        초기화 함수. label을 index로 변환하기 위한 딕셔너리 형태인 labels_map을 만들어야 한다.
        """
        # 총 30개의 index를 분류한다. 하지만 test set에서는 아래 10개 이외의 것은 전부 extra로 한다.
        #yes, no, up, down, left, right, on, off, stop, go를 각각의 index로 부여한다
        # index로 변환하기 위한 함수
        files = {'yes':0,'no':1,'up':2,'down':3,'left':4,'right':5,'on':6,'off':7,'stop':8,'go':9, 'bed':10,'bird':11,'cat':12,'dog':13,'eight':14,
        'five':15,'four':16,'happy':17,'house':18,'marvin':19,'nine':20,'one':21,'seven':22,'shella':23,'six':24,'three':25,'tree':26,'two':27,'wow':28,'zero':29,
        }
        keys = list(self.files.keys())
        self.file_list = [] # 실제 wav 파일의 각 location들이 들어있는 리스트
        self.file_index = [] # file_list의 순서대로 어떠한 음인지, index가 들어있다.
        self.size = 0

        if(train):
            for elem in keys:
                tmp_path = os.path.join(manifest_filepath,elem)
                tmp_path += '/*'
                self.file_list += glob.glob(tmp_path)[0:int(len(glob.glob(tmp_path))*0.7)]
                self.file_index += [elem]*len(glob.glob(tmp_path)[0:int(len(glob.glob(tmp_path))*0.7)])
                self.size += len(glob.glob(tmp_path)[0:int(len(glob.glob(tmp_path))*0.7)])
        else:
            for elem in keys:
                tmp_path = os.path.join(manifest_filepath,elem)
                tmp_path += '/*'
                self.file_list += glob.glob(tmp_path)[int(len(glob.glob(tmp_path))*0.7):]
                self.file_index += [elem] * len(glob.glob(tmp_path)[int(len(glob.glob(tmp_path))*0.7):])
                self.size += len(glob.glob(tmp_path)[int(len(glob.glob(tmp_path))*0.7):])

        super(SpectogramDataset, self).__init__(audio_conf)

    def __getitem__(self, index):

        return self.parse_audio(self.file_list[index]), self.file_index[index]


    def __len__(self):
        return self.size



class AudioDataLoader(DataLoader):
    """
    DataLoader.
    초기화 input:
        dataset : spectogramdataset이 될 것
        minibatch size : default might be 20
        worker : loading을 위한 process의 갯수.
    """
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def _collate_fn(batch):
    """
    현재 간헐적으로 길이가 맞지 않는 놈들이 있다.
    이놈들을 161 X 101의 FloatTensor로 만들기 위한 collate_fn
    input:
        batch - List of dataset. dataset에서 샘플링한 것의 리스트이다.
    output -  list. FloatTensor, list of labels
    """
    minibatch_size = len(batch)
    inputs = torch.zeros(minibatch_size, 161, 101) # N X H X W
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
        targets.append(target)
    return [inputs, targets]
