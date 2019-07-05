"""test.py
Test the model stored at `model_path` to see how effectively it predicts
the final distribution of states given the initial state.
"""
import pickle
import configparser
import utils
import torch

from predict_model import Controller
from dataset import Dataset


def main():
    args = utils.parse_args()
    model_path = "log/simulate_crossing/predictor_2/model_m_0.tar"
    data_path = "barge_in_final_states_test.p"

    f = open(data_path, "rb")
    data = Dataset(pickle.load(f))

    model_config = configparser.RawConfigParser()
    model_config.read(args.model_config)
    model = Controller(model_config,
                       model_type=args.model_type)  # model_type = crossing
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    with torch.no_grad():
        for example in data:
            batch_of_one = example[0].unsqueeze(0)
            pred = utils.get_coefs(model(batch_of_one))
            actual = example[1]
            print("Prediction: {}, Actual: {}".format(pred, actual))


if __name__ == "__main__":
    main()
