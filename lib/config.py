class classConfig:
    def __init__(self):
        # 언어 데이터 파일 경로
        # 言語DATAのPATH
        self.LANGUAGE_FILE_PATH = "./simte.txt"

        # 언어 데이터 사용 범위(크기)
        # 言語DATAの使用範囲（SIZE）
        self.USE_DATA_COUNT = 19

        self.MODEL_FILE_PATH = './testModel.h5'

        self.EPOCHS = 2000


        # 태그 단어
        # シンボル単語
        self.PAD = "<P>"
        self.START = "<S>"
        self.END = "<E>"
        self.OOV = "<O>"  # Out of Vocabulary

        # 태그 인덱스
        #　シンボル・インデックス
        self.START_INDEX = 1
        self.END_INDEX = 2
        self.OOV_INDEX = 3

        # 데이터 타입
        #　DATAタイプ
        self.ENCODER_INPUT = 0
        self.DECODER_INPUT = 1
        self.DECODER_TARGET = 2

        # Embedding Vector Dimension
        self.EMBEDDING_DIM = 100

        # LSTM Hidden Layer Dimension
        self.LSTM_HIDDEN_DIM = 128

