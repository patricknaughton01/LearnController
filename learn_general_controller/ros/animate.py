import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import animation 
from utils import parse_args, covariance_to_std

def play(ground_truth, seq_lengths, pred=None, var=None, model_uncertainty=None, data_uncertainty=None, num=0, video_name=""):
    print("Current sequence length: ", seq_lengths[num])
    # create figure object
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    plt.rc('font', size=8)  
    
    seq_length = seq_lengths[num]
    last_frame = seq_length
    episode = ground_truth
    n = int((episode.shape[1] - 4) / 5)
    colors = [np.random.rand(3) for _ in range(n)]
    
    min_x = np.min(ground_truth[:last_frame, 0]) - 2
    max_x = np.max(ground_truth[:last_frame, 0]) + 2

    min_y = np.min(ground_truth[:last_frame, 1]) - 2
    max_y = np.max(ground_truth[:last_frame, 1]) + 2

    text_offset = 0.5
    # set title for the figure
    fig.suptitle('Animation', fontsize=20)

    # Function: define the initial state of this figure object
    def init():  
        ax.plot([],[]) 
        return ax

    # Function: draw ith frame
    def animate(i):

        # clear the last frame
        plt.hold(False)	
        
        end_idx = min(i+1, last_frame)
        start_idx = max(min(i-1, end_idx-1), 0)

        ax.plot(pred[:last_frame, num, 0], pred[:last_frame, num, 1], 'k-o', markersize=5)
        plt.hold(True)
        ax.plot(pred[start_idx:end_idx, num, 0], pred[start_idx:end_idx, num, 1], 'r-o')
        
        
        for i in range(n):
            if i == 0:
                x_idx, y_idx = 0, 1
                radius = episode[0, 4]
            else:
                x_idx, y_idx = 9 + (i-1) * 5, 9 + (i-1) * 5 + 1
                radius = episode[0, 13 + (i-1) * 5]
                ax.plot(episode[start_idx:end_idx, x_idx], episode[start_idx:end_idx, y_idx], 'r-o')

            plt.plot(episode[:, x_idx], episode[:, y_idx], '-*', color=colors[i])  
            
            # e = patches.Ellipse((episode[t, x_idx], episode[t, y_idx]), radius, radius, linewidth=2, fill=False, zorder=2, color=colors[i])
            # ax.add_patch(e)

        if var is not None:
            xcenter1, ycenter1 = pred[start_idx, num, 0], pred[start_idx, num, 1]
            std_x1, std_y1 = var[start_idx, num, 0], var[start_idx, num, 1]
            print('std x: %.4f, std y: %.4f' % (std_x1, std_y1))
            e1 = patches.Ellipse((xcenter1, ycenter1), std_x1, std_y1, linewidth=2, fill=False, zorder=2)
            ax.add_patch(e1)

        if model_uncertainty is not None and data_uncertainty is not None:
            print("model std x: %.4f, model std y: %.4f" % (model_uncertainty[start_idx, num, 0], model_uncertainty[start_idx, num, 1]))
            print("data std x: %.4f, data std y: %.4f" % (data_uncertainty[start_idx, num, 0], data_uncertainty[start_idx, num, 1]))

        # set xlim, ylim, zlim
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.axis('equal')
        return ax

    # Start playing!!!	
    frame_count = pred.shape[0] if pred is not None else seq_lengths[num]
    anim = animation.FuncAnimation(fig, animate, init_func=init,  frames=frame_count, interval=1000) 
    if video_name:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(video_name, writer=writer)
    else:
        plt.show()

def main():
    args = parse_args()
    parent_path = "log/%s/seed_%d_bootstrap_%s_lstm_%d_M_%d_length_scale_%.4f_T_%d_tao_%.4f_dropout_%.4f_exp_%d" % (
                args.train_data_name,
                args.seed, 
                str(args.bootstrap), 
                args.num_lstms, 
                args.M, 
                args.length_scale, 
                args.T, 
                args.tao, 
                args.dropout, 
                args.exp) 
    parent_path = parent_path + "/test/%s" % args.test_data_name
    
    print("Data path: %s" % parent_path)
    
    val_states = np.load(parent_path + '/val_states.npy')
    val_seq_lengths = np.load(parent_path + '/val_seq_lengths.npy')
    val_pred = np.load(parent_path + '/val_pred_x.npy')
    
    val_var = np.load(parent_path + '/val_var_x.npy')
    val_data_uncertainty = np.load(parent_path + '/val_data_uncertainty.npy')
    val_model_uncertainty = np.load(parent_path + '/val_model_uncertainty.npy')

    val_var = covariance_to_std(val_var)
    val_data_uncertainty = covariance_to_std(val_data_uncertainty)
    val_model_uncertainty = covariance_to_std(val_model_uncertainty)

    play(val_states[args.num], val_seq_lengths, 
    pred=val_pred, 
    var=val_var, 
    model_uncertainty=val_model_uncertainty,
    data_uncertainty=val_data_uncertainty,
    num=args.num,
    video_name=args.video_name)  

if __name__ == '__main__':
    main()