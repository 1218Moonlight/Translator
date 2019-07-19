from lib import preprocess, machine, config
import numpy as np
import os

class classBox(config.classConfig):
    def __init__(self):
        super().__init__()
        self.pre = preprocess.classPre()
        self.learn = machine.classModel()
        self.__preprocess()
        self.__machine()

    def cli(self):
        while True:
            msg = input('input : ')
            if msg == 'exit':
                break
            inputSeq = self.__makePredictInput(msg)
            print('[ inputSeq ]\n : {}'.format(inputSeq))
            sentence = self.__generateText(inputSeq)
            print('[ sentence ]\n : {}'.format(sentence))

    def __preprocess(self):
        self.src, self.tar = self.pre.readLanguageFile()
        self.words = self.pre.oneData(self.src, self.tar)
        self.wordToIndex, self.indexToWord = self.pre.indexingData(self.words)
        self.max_len = max([len(line) for line in self.words])
        self.wordsSize = len(self.words)

        self.xEncoder = self.pre.convertTextToIndex(self.src, self.wordToIndex, self.pre.ENCODER_INPUT,
                                                      self.max_len)
        self.xDecoder = self.pre.convertTextToIndex(self.tar, self.wordToIndex, self.pre.DECODER_INPUT,
                                                      self.max_len)
        self.yDecoder = self.pre.convertTextToIndex(self.tar, self.wordToIndex, self.pre.DECODER_TARGET,
                                                      self.max_len)

        # 학습시 입력은 인덱스이지만, 출력은 원핫인코딩 형식임
        # トレーニング時の入力はインデックスだけど、出力はOneHotEncodingの形式である。
        self.yDecoder = self.pre.oneHotEncoding(self.yDecoder, self.max_len, self.wordsSize)

    def __machine(self):
        self.trainingEncoderBox = self.learn.definitionTrainingModelEncoder(self.wordsSize)
        self.trainingDecoderBox = self.learn.definitionTrainingModelDecoder(self.wordsSize,
                                                                              self.trainingEncoderBox.get(
                                                                                  'encoderStates'))

        self.model = self.learn.definitionTrainingModel(self.trainingEncoderBox.get('encoderInputs'),
                                                          self.trainingDecoderBox.get('decoderInputs'),
                                                          self.trainingDecoderBox.get('decoderOutputs'))

        self.encoderModel = self.learn.definitionPredictiveModelEncoder(self.trainingEncoderBox.get('encoderInputs'),
                                                                          self.trainingEncoderBox.get('encoderStates'))

        self.decoderModel = self.learn.definitionPredictiveModelDecoder(
            self.trainingDecoderBox.get('decoderEmbedding'),
            self.trainingDecoderBox.get('decoderInputs'),
            self.trainingDecoderBox.get('decoderLstm'),
            self.trainingDecoderBox.get('decoderDense'))

        self.isModelFile = os.path.isfile(self.learn.MODEL_FILE_PATH)

        if self.isModelFile:
            self.model.load_weights(self.learn.MODEL_FILE_PATH)
        else:
            self.learn.training(self.model, self.xEncoder, self.xDecoder, self.yDecoder)
            self.model.save_weights(self.learn.MODEL_FILE_PATH)

    # 예측을 위한 입력 생성
    # 予測の為の入力生成
    def __makePredictInput(self, sentence):
        sentences = []
        sentences.append(sentence)
        return self.pre.convertTextToIndex(sentences, self.wordToIndex, self.learn.ENCODER_INPUT, self.max_len)

    # 텍스트 생성
    # テキスト生成
    def __generateText(self, inputSeq):
        states = self.encoderModel.predict(inputSeq)
        targetSeq = np.zeros((1, 1))
        targetSeq[0, 0] = self.START_INDEX
        indexs = []
        while 1:
            decoderOutputs, state_h, state_c = self.decoderModel.predict(
                [targetSeq] + states)
            index = np.argmax(decoderOutputs[0, 0, :])
            indexs.append(index)
            if index == self.END_INDEX or len(indexs) >= self.max_len:
                break
            targetSeq = np.zeros((1, 1))
            targetSeq[0, 0] = index
            states = [state_h, state_c]
        sentence = self.pre.convertIndexToText(indexs, self.indexToWord)
        return sentence
