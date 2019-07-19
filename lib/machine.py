from keras import models
from keras import layers
from lib import config


class classModel(config.classConfig):

    # 훈련 모델 인코더 정의
    # 訓練モデルエンコーダ定義
    def definitionTrainingModelEncoder(self, wordsSize):
        encoderInputs = layers.Input(shape=(None,))
        # Embedding Layer
        encoderOutputs = layers.Embedding(wordsSize, self.EMBEDDING_DIM)(encoderInputs)
        encoderOutputs, state_h, state_c = layers.LSTM(self.LSTM_HIDDEN_DIM,
                                                       dropout=0.1,
                                                       recurrent_dropout=0.5,
                                                       return_state=True)(encoderOutputs)
        # Hidden State와 Cell State를 하나로...
        # Hidden StateとCell Stateを一つに…
        encoderStates = [state_h, state_c]
        return {'encoderStates': encoderStates, 'encoderInputs': encoderInputs}

    # 훈련 모델 디코더 정의
    # 訓練モデルディコーダ定義
    def definitionTrainingModelDecoder(self, wordsSize, encoderStates):
        decoderInputs = layers.Input(shape=(None,))
        decoderEmbedding = layers.Embedding(wordsSize, self.EMBEDDING_DIM)
        decoderOutputs = decoderEmbedding(decoderInputs)
        # encoder와는 다르게 return_sequences를 True로 설정          # return_sequencesをTrueに設定
        # 모든 타임 스텝 출력값 리턴!                                # すべてのタイム・ステップ出力をリターン！
        # 출력값들을 다음 Layer의 Dense()로 처리.                    # 出力のValuesは次のLayerのDense()で処理。
        decoderLstm = layers.LSTM(self.LSTM_HIDDEN_DIM,
                                  dropout=0.1,
                                  recurrent_dropout=0.5,
                                  return_state=True,
                                  return_sequences=True)
        # initial_state를 encoder의 state로 초기화
        # initial_stateをEncoderのStateで初期化
        decoderOutputs, _, _ = decoderLstm(decoderOutputs,
                                           initial_state=encoderStates)
        decoder_dense = layers.Dense(wordsSize, activation='softmax')
        decoderOutputs = decoder_dense(decoderOutputs)
        return {'decoderOutputs': decoderOutputs, 'decoderInputs': decoderInputs,
                'decoderEmbedding': decoderEmbedding, 'decoderLstm': decoderLstm,
                'decoderDense': decoder_dense}

    # 훈련 모델 정의
    # 訓練モデル定義
    def definitionTrainingModel(self, encoderInputs, decoderInputs, decoderOutputs):
        return models.Model(input=[encoderInputs, decoderInputs], output=decoderOutputs)

    # 예측 모델 인코더 정의
    # 予測モデルエンコーダ定義
    def definitionPredictiveModelEncoder(self, encoderInputs, encoderStates):
        return models.Model(encoderInputs, encoderStates)

    # 예측 모델 디코더 정의
    # 予測モデルディコーダ定義
    def definitionPredictiveModelDecoder(self, decoderEmbedding, decoderInputs, decoderLstm, decoderDense):
        # 예측시에는 훈련시와 달리 타임 스텝을 한 단계씩 수행
        # 매번 이전 디코더 상태를 입력으로 받아서 새로 설정
        # 予測時は、訓練時とは違く、タイム・ステップを一段階ずつ遂行。
        # 毎回、以前Decoder Stateを入力で受け取り新たに設定
        decoderStateInput_h = layers.Input(shape=(self.LSTM_HIDDEN_DIM,))
        decoderStateInput_c = layers.Input(shape=(self.LSTM_HIDDEN_DIM,))
        decoderStatesInputs = [decoderStateInput_h, decoderStateInput_c]
        decoderOutputs = decoderEmbedding(decoderInputs)
        # LSTM Layer
        decoderOutputs, state_h, state_c = decoderLstm(decoderOutputs,
                                                       initial_state=decoderStatesInputs)
        decoderStates = [state_h, state_c]
        # Dense Layer를 통해 OneHotEncodding 형식으로 각 단어 인덱스를 출력
        # Dense Layerを通じてOneHotEncodding形式で各の単語インデクスを出力
        decoderOutputs = decoderDense(decoderOutputs)
        decoderModel = models.Model([decoderInputs] + decoderStatesInputs,
                                    [decoderOutputs] + decoderStates)
        return decoderModel

    def training(self, model, xEncoder, xDecoder, yDecoder):
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x=[xEncoder, xDecoder],
                  y=yDecoder,
                  epochs=self.EPOCHS,
                  batch_size=64,
                  verbose=2)
