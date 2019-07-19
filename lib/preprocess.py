import pandas as pd
import numpy as np
from lib import config
from keras.preprocessing.sequence import pad_sequences


class classPre(config.classConfig):

    # 언어 데이터 로드 후, 사용 범위 지정 및 source와 target으로 나누기.
    # 言語DATAをLOADの後、使用範囲及びSourceとTargetに分ける。
    def readLanguageFile(self):
        lines = pd.read_csv(self.LANGUAGE_FILE_PATH, encoding='utf-8', names=['src', 'tar'])
        lines = lines[0:self.USE_DATA_COUNT]
        src, tar = list(lines.src), list(lines.tar)
        return src, tar

    # 언어 데이터 및 심볼 통합
    # 言語DATA及びシンボル統合
    def oneData(self, src, tar):
        sentences = []
        sentences.extend(src)
        sentences.extend(tar)
        words = []
        for sentence in sentences:
            for word in sentence.split():
                words.append(word)
        # # 길이가 0인 단어는 삭제
        # # SIZEが０の単語を削除
        # words = [word for word in words if len(word) > 0]
        #
        # # 중복 제거
        # # 重複削除
        # words = list(set(words))
        words[:0] = [self.PAD, self.START, self.END, self.OOV]
        return words

    # word, index의 dictionary 생성
    # word, indexの dictinary 生成
    def indexingData(self, words):
        wordToIndex = {word: index for index, word in enumerate(words)}
        indexToWord = {index: word for index, word in enumerate(words)}
        return wordToIndex, indexToWord

    # 문장을 인덱스로 변환
    # 文章をインデックスに変換
    def convertTextToIndex(self, sentences, vocabulary, type, max_len):
        if type == self.DECODER_INPUT:
            # sentences = [self.START + " " + m + " " + self.END for m in sentences]
            sentences = [self.START + " " + m for m in sentences]
        elif type == self.DECODER_TARGET:
            sentences = [m + " " + self.END for m in sentences]

        sentencesDic = []
        for sentence in sentences:
            sentenceDic = []
            for word in sentence.split():
                if vocabulary.get(word) is not None:
                    # 사전에 있는 단어면 해당 인덱스를 추가
                    # 辞書にある単語ならばインデックスを追加。
                    sentenceDic.extend([vocabulary[word]])
                else:
                    # 사전에 없는 단어면 OOV 인덱스를 추가
                    # 辞書にない単語ならばOOVを追加。
                    sentenceDic.extend([vocabulary[self.OOV]])
            sentencesDic.append(sentenceDic)
        # padding
        sentencesDic = pad_sequences(sentencesDic, maxlen=max_len, padding='post')
        return np.asarray(sentencesDic)

    def oneHotEncoding(self, xy, max_len, wordsSize):
        # initialization
        oneHotData = np.zeros((len(xy), max_len, wordsSize))
        for i, sequence in enumerate(xy):
            for j, index in enumerate(sequence):
                oneHotData[i, j, index] = 1
        return oneHotData

    # 인덱스를 문장으로 변환
    # インデクスを文章に変換
    def convertIndexToText(self, indexs, vocabulary):
        sentence = ''

        for index in indexs:
            if index == self.END_INDEX:
                break
            if vocabulary.get(index) is not None:
                sentence += vocabulary[index]
            else:
                sentence.extend([vocabulary[self.OOV_INDEX]])
            # Add Blank
            sentence += ' '
        return sentence
