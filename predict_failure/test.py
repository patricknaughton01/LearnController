"""test.py
Test the model stored at `model_path` to see how effectively it predicts
the final distribution of states given the initial state.
"""
import pickle
import configparser
import utils
import torch
import matplotlib.pyplot as plt

from predict_model import Controller
from dataset import Dataset


def main():
    args = utils.parse_args()
    model_path = args.model_path
    data_path = args.data_path
    if model_path == "" or data_path == "":
        print("You must specify both a model_path and data_path")
        return

    f = open(data_path, "rb")
    data = Dataset(pickle.load(f))

    out = open("test_output.txt", "w")
    #out.write("mux\tmuy\tsx\tsy\tcorr\tax\tay\n")

    model_config = configparser.RawConfigParser()
    model_config.read(args.model_config)
    model = Controller(model_config,
                       model_type=args.model_type)  # model_type = crossing
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for example in data:
            batch_of_one = example[0].unsqueeze(0)
            pred = model(batch_of_one)
            actual = example[1]
            loss, _ = utils.neg_2d_gaussian_likelihood(pred,
                                                       actual.unsqueeze(0))
            total_loss += loss.item()
            #print("Prediction: {}, Actual: {}".format(pred, actual))
            coefs = utils.get_coefs(pred)
            for t in coefs:
                out.write(str(t.item()) + "\t")
            for val in actual:
                out.write(str(val.item()) + "\t")
            out.write("\n")
            out.write(str(example[2]) + "\n")
            out.write(str(example[3]) + "\n")
    print("Average loss: ", total_loss / len(data))



if __name__ == "__main__":
    main()
