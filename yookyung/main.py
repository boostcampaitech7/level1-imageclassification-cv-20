import argparse
import torch
import train  # train.py 모듈 불러오기
import test  # test.py 모듈 불러오기
from config import config

# 장비 설정
device = torch.device(config.DEVICE)

def main():
    parser = argparse.ArgumentParser(description="Train and Test the model")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'all'], help="Choose mode: 'train', 'test', or 'all'")
    
    args = parser.parse_args()

    # 학습만 진행
    if args.mode == 'train' or args.mode == 'all':
        print("Starting training process...")
        train.train(device)  # train.py의 train 함수 실행

    # 추론만 진행
    if args.mode == 'test' or args.mode == 'all':
        print("Starting inference process...")
        test.test(device)  # test.py의 test 함수 실행

if __name__ == "__main__":
    main()
