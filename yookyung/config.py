class Config:
    # 데이터 관련
    TRAIN_DATA_DIR = "./yookyung/data/train"
    TEST_DATA_DIR = './yookyung/data/test'

    TRAIN_RATIO = 0.8
    BATCH_SIZE = 64
    IMAGE_SIZE = (224, 224)
    
    # 모델 관련
    MODEL_TYPE = 'timm' # torchvision, timm
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    NUM_CLASSES = 500
    
    # 훈련 관련
    EPOCHS = 50
    LEARNING_RATE = 0.001
    OPTIMIZER = "adam"

    # 손실함수 관련
    LOSS = "cross_entropy" # cross_entropy, bce_with_logits, mse
    LOSS_PARAMS = {
        "cross_entropy": {},  # CrossEntropyLoss는 기본 파라미터 사용
        "bce_with_logits": {"pos_weight": None},  # BCEWithLogitsLoss 사용 시 설정
        "mse": {},  # MSELoss 사용 시 설정
        # 필요한 다른 손실 함수들 추가
    }

    # 스케쥴러 관련
    SCHEDULER = "cosine" # cosine, step
    SCHEDULER_PARAMS = {
        "step": {
            "step_size": 2,  # epochs
            "gamma": 0.1
        },
        "cosine": {
            "T_max": 10,  # 전체 주기
            "eta_min": 1e-6  # 최소 학습률
        }
    }

    # 하드웨어 및 환경
    DEVICE = "cuda"
    SEED = 20
    
    # 로깅 및 체크포인트
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = f"./yookyung/train_result/{MODEL_NAME}"
    SAVE_TOP_K = 3  # 저장할 최상위 모델 수
    
    # 평가 관련
    EVAL_METRIC = "accuracy"
    VALID_INTERVAL = 1
    
    # 추론 관련
    INFERENCE_MODEL_PATH = "./yookyung/train_result/best_model.pth"
    RESULT_DIR = "./yookyung/output"

config = Config()