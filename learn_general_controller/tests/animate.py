import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import patches
from matplotlib import animation 
from glob import glob

def play(episode, obstacles=None, video_name=None):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    plt.rc('font', size=8)  
    
    last_frame = episode.shape[0]
    n = int((episode.shape[1] - 4) / 5)
    print('number of agents: %d' % n)
    print('episode shape: ', episode.shape)
    colors = [np.random.rand(3) for _ in range(n)]
    
    min_x = np.min(episode[:, 0]) - 2
    max_x = np.max(episode[:, 0]) + 2

    min_y = np.min(episode[:, 1]) - 2
    max_y = np.max(episode[:, 1]) + 2

    # set title for the figure
    fig.suptitle('Animation', fontsize=20)

    # Function: define the initial state of this figure object
    def init():  
        ax.plot([],[]) 
        return ax

    # Function: draw ith frame
    def animate(ith_frame):

        # clear the last frame
        plt.hold(False)	
        
        end_idx = min(ith_frame+1, last_frame)
        start_idx = max(min(ith_frame-1, end_idx-1), 0)

        for i in range(n):
            if i == 0:
                x_idx, y_idx = 0, 1
                radius = episode[0, 4]
            else:
                x_idx, y_idx = 9 + (i-1) * 5, 9 + (i-1) * 5 + 1
                radius = episode[0, 13 + (i-1) * 5]
            plt.plot(episode[start_idx:end_idx, x_idx], episode[start_idx:end_idx, y_idx], 'r-o')
            plt.hold(True)

            e = patches.Ellipse((episode[ith_frame, x_idx], episode[ith_frame, y_idx]), radius * 2, radius * 2, linewidth=2, fill=False, zorder=2, color=colors[i])
            ax.add_patch(e) 
            plt.plot(episode[:, x_idx], episode[:, y_idx], '-*', color=colors[i])  
            
            # e = patches.Ellipse((episode[t, x_idx], episode[t, y_idx]), radius, radius, linewidth=2, fill=False, zorder=2, color=colors[i])
            # ax.add_patch(e)
        if obstacles is not None:
            num_vertices = int(obstacles.shape[1] / 2)
            num_obstacles = obstacles.shape[0]
            for j in range(num_obstacles):
                for k in range(num_vertices):
                    plt.plot(obstacles[j, [k * 2, (k * 2 + 2) % obstacles.shape[1]]], 
                             obstacles[j, [1 + 2 * k, (3 + 2 * k) % obstacles.shape[1]]], 'k-')
        
        # set xlim, ylim, zlim
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.axis('equal')
        return ax

    # Start playing!!!	
    anim = animation.FuncAnimation(fig, animate, init_func=init,  frames=episode.shape[0], interval=1000) 
    if video_name:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(video_name, writer=writer)
    else:
        plt.show()

if __name__ == "__main__":

    # PATH = '/Users/lingyaozhang/study/research/learn_general_controller/data/simulate_overtaking'
    PATH = '/home/yashoza/Downloads/LearnControllers/learn_general_controller/data/simulate_overtaking'
    # OBSTACLE_PATH = '/Users/lingyaozhang/study/research/RVO2/examples/data/Barge_in'
    NUM = 300
    VIDEO_NAME = None
    # VIDEO_NAME = 'New_Four_People_Avoiding_Horizontal_Scenario2.mp4'
    episode = np.load(PATH + '/states.npy')[NUM]
    # obstacle_files = glob(OBSTACLE_PATH + "_obstacles_*.txt")
    # obstacle_path = OBSTACLE_PATH + "_obstacles_%d.txt" % NUM
    # print(obstacle_path)
    # if obstacle_path in obstacle_files:
    #     obstacles = np.loadtxt(obstacle_path)
    #     play(episode, obstacles, video_name=VIDEO_NAME)
    # else:
    #     play(episode)
    play(episode, video_name=VIDEO_NAME)
    