import numpy as np
from utils import parse_args
from glob import glob
from model import *
import os
from utils import *
import configparser
from time import time

def main():
    args = parse_args()
    config = vars(args)
    parent_path = "log/%s/seed_%d_bootstrap_%s_M_%d" % (
                args.train_data_name,
                args.seed,
                str(args.bootstrap),
                args.M)
    print('parent path is %s' % parent_path)
    save_path = parent_path + "/test/%s" % args.test_data_name
    if args.show_mc:
        save_path = save_path + "/mc_dropout"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("save path is %s" % save_path)
    checkpoint_path = parent_path + "/model_m_*.tar"

    data_path = args.data_path + '/' + args.test_data_name
    all_states = np.load(data_path + '/states.npy', allow_pickle=True)
    #print(all_states)
    all_seq_lengths = np.load(data_path+ '/seq_lengths.npy', allow_pickle=True)
    all_num_humans = np.load(data_path + '/num_humans.npy', allow_pickle=True)
    val_loader = lambda: dataloader((all_states, all_seq_lengths, all_num_humans), config, is_train=False, shuffle=False)

    models = []
    checkpoint_file_names = glob(checkpoint_path)

    model_config = configparser.RawConfigParser()
    model_config.read(args.model_config)
    for checkpoint_file_name in checkpoint_file_names:
        model = Controller(model_config, model_type=args.model_type)
        if os.path.isfile(checkpoint_file_name):
            checkpoint = torch.load(checkpoint_file_name)
            model.load_state_dict(checkpoint['state_dict'])
            print('Loaded checkpoint: %s' % checkpoint_file_name)
            models.append(model)
        else:
            raise ValueError('No such checkpoint: %s' % checkpoint_file_name)
        if args.show_mc:
            model.train()
        else:
            model.eval()
    criterion = neg_2d_gaussian_likelihood
    val_loss = []
    val_l2_error = []

    max_seq_len = np.max(all_seq_lengths) - 1
    total_size = len(all_seq_lengths)
    state_dim = all_states[0].shape[1]
    final_states = np.empty((total_size, ), dtype=object)
    final_seq_lengths = np.zeros((total_size, ))
    final_pred_xs = np.zeros((max_seq_len, total_size, 2))
    final_var_xs = np.zeros((max_seq_len, total_size, 2, 2))
    final_data_uncertainty = np.zeros((max_seq_len, total_size, 2, 2))
    final_model_uncertainty = np.zeros((max_seq_len, total_size, 2, 2))
    start_idx = 0
    for batch_idx, (states, seq_lengths, targets, future_states) in enumerate(val_loader(), 1):
        # print("states shape", states.shape)
        val_preds = []
        val_pred_xs = []
        seq_lengths = torch.from_numpy(seq_lengths).long()
        targets = torch.from_numpy(targets).float()
        model_num = 0

        flag_new_pred = 0

        for model in models:
            # states: seq_len x batch_size x dim # for the first state - (29, 8, 39)
            # seq_len = no of points in the path
            # batch_size = just batch size
            # dim = number of "types" of data

            # print("eval model", model_num )
            # model_num+=1

            seq_len = states.shape[0]
            # print("states.shape", states.shape) # (29, 8, 39)
            outputs = []
            pred_xs = []
            h_t = None
            for i in range(seq_len):
                # print("i is ", i)
                cur_states = states[i] # size (8,39)

                # if flag_new_pred is 1:
                if i > 0:
                    # print("adding new data now ")
                    cur_states[:, 0:2] = (new_pred.data).cpu().numpy() # (Variable(x).data).cpu().numpy()

                # if batch_idx > 1:
                #     # take the prev state and add the new predicted state to the last element

                # print("cur_states shape is ", cur_states) # (8,39)
                cur_rotated_states = transform_and_rotate(cur_states) # Size([8, 6, 13])
                # print("cur_rotated_states", cur_rotated_states)

                # now state_t is of size: batch_size x num_human x dim
                batch_size = cur_states.shape[0]
                # print("batch_size", batch_size)

                batch_occupancy_map = []

                start_time = time()
                for b in range(batch_size):
                    occupancy_map = build_occupancy_maps(build_humans(cur_states[b]))
                    batch_occupancy_map.append(occupancy_map)

                batch_occupancy_map = torch.stack(batch_occupancy_map)#[:,
                # 1:, :]
                state_t = torch.cat([cur_rotated_states, batch_occupancy_map], dim=-1) # Size([8, 6, 61])
                print(state_t.size())
                # print("state_t.shape", state_t.shape)
                # print("cur_rotated_states.shape", cur_rotated_states.shape)
                # print("batch_occupancy_map.shape", batch_occupancy_map.shape)

                pred_t, h_t = model(state_t, h_t)
                # print("the pred_t coming out of the network is ", pred_t)
                outputs.append(pred_t)
                new_pred = torch.from_numpy(cur_states[:, 0:2]).float() + pred_t[:, 0:2]
                flag_new_pred = 1
                pred_xs.append(torch.from_numpy(cur_states[:, 0:2]).float() + pred_t[:, 0:2])
                # print("the pred_xs coming out of the network is ", (torch.from_numpy(cur_states[:, 0:2]).float() + pred_t[:, 0:2]))
                # print("pred_t.shape", (pred_t[:, 0:2])) # constant 8
                # print("pred_xs.shape", len(pred_xs)) # inc with i
                end_time = time()
                # print('Average prediction time for this batch is : %.4f seconds' % ((end_time - start_time) / batch_size))

            outputs = torch.stack(outputs)
            pred_xs = torch.stack(pred_xs)

            loss, l2_error = criterion(outputs, targets, seq_lengths)
            val_loss.append(loss.item())
            val_l2_error.append(l2_error.item())

            val_preds.append(outputs)
            val_pred_xs.append(pred_xs)
        val_pred_x, val_var_x, val_data_uncertainty, val_model_uncertainty = ensemble(val_preds, val_pred_xs)
        end_idx = start_idx + len(seq_lengths)
        cur_max_seq_len = future_states.shape[0]
        final_seq_lengths[start_idx:end_idx] = seq_lengths.cpu().numpy()
        for i, seq_len in enumerate(seq_lengths):
            final_states[start_idx + i] = future_states[0:seq_len, i, :]
        final_pred_xs[0:cur_max_seq_len, start_idx:end_idx, :] = val_pred_x
        # print("val_pred_x is", val_pred_x)
        final_var_xs[0:cur_max_seq_len, start_idx:end_idx, :] = val_var_x
        final_data_uncertainty[0:cur_max_seq_len, start_idx:end_idx, :] = val_data_uncertainty
        final_model_uncertainty[0:cur_max_seq_len, start_idx:end_idx, :] = val_model_uncertainty
        start_idx = end_idx

    print('Data is now saved in: %s' % save_path)
    np.save(save_path + "/val_states.npy", final_states)
    np.save(save_path + "/val_seq_lengths.npy", final_seq_lengths.astype(int))
    np.save(save_path + "/val_pred_x.npy", final_pred_xs)
    np.save(save_path + "/val_var_x.npy", final_var_xs)
    np.save(save_path + "/val_data_uncertainty.npy", final_data_uncertainty)
    np.save(save_path + "/val_model_uncertainty.npy", final_model_uncertainty)
    print("avg loss: %.4f" % np.mean(val_loss))
    print("avg l2 error: %.4f" % np.mean(val_l2_error))

if __name__ == "__main__":
    main()
