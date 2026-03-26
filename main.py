from src.train import train_model
from src.data_loader import get_generators
from src.evaluate import evaluate_model
from src.inference import predict_folder

def main():

    model, history = train_model()

    train_gen, val_gen, test_gen = get_generators()

    threshold = evaluate_model(model, val_gen, test_gen)

    # test folder riêng
    predict_folder(model, "your_test_images", threshold)


if __name__ == "__main__":
    main()
