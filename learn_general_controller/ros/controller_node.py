#!/usr/bin/env python
# import rospy
# import message_filters
# from spencer_tracking_msgs.msg import TrackedPersons # Right import??
# from humanoid_nav_msgs.msg import controller_output
# from nav_msgs.msg import GridCells, Odometry
import argparse
from utils import build_occupancy_maps, transform_and_rotate, build_humans, ensemble
from glob import glob
from model import *
import torch
import numpy as np
import os 

import configparser

PED_TOPIC_NAME = '/pedsim/tracked_persons'
OBSTACLE_TOPIC_NAME = '/pedsim/static_obstacles'
ROBOT_TOPIC_NAME = '/pedsim/robot_position'

RADIUS = 0.2
GOAL_POS_X = 0.0
GOAL_POS_Y = 0.0
PREF_SPEED = 1.5

h_ts = [None for _ in range(5)] 
pub = None
models = []

def parse_args():
    parser = argparse.ArgumentParser(description="Learn Controller")
    parser.add_argument("--controller", type=str, default="crossing")
    parser.add_argument("--bootstrap", action='store_true')
    parser.add_argument("--mc_dropout", action='store_true')
    parser.add_argument("--model_config", type=str, default="configs/model.config")
    
    return parser.parse_args()

def closest_point_on_wall(wall_msg, robot_msg):
  wall_radius = np.sqrt(wall_msg.cell_width ** 2 + wall_msg.cell_height ** 2)

  min_dist = float('inf')
  closest_cell = None
  for cell in wall_msg.cells:
    cur_dist = (robot_msg.pose.pose.position.x - cell.x) ** 2 + (robot_msg.pose.pose.position.y - cell.y) ** 2
    if cur_dist < min_dist:
      min_dist = cur_dist
      closest_cell = cell 
  return closest_cell, wall_radius

def callback():
  global h_ts
  # theta = np.arctan2(robot_msg.twist.twist.linear.y, 
  #                    robot_msg.twist.twist.linear.x)
  # features = [robot_msg.pose.pose.position.x,
  #             robot_msg.pose.pose.position.y,
  #             robot_msg.twist.twist.linear.x,
  #             robot_msg.twist.twist.linear.y,
  #             RADIUS,
  #             GOAL_POS_X, # How do I get the goal position?
  #             GOAL_POS_Y,
  #             PREF_SPEED,
  #             theta]

  # for tracker_person in ped_msg.tracks:
  #   features.extend([tracker_person.pose.pose.position.x,
  #                    tracker_person.pose.pose.position.y,
  #                    tracker_person.twist.twist.linear.x,
  #                    tracker_person.twist.twist.linear.y,
  #                    RADIUS]) # Why would I need track_id??

  # closest_cell, wall_radius = closest_point_on_wall(wall_msg, robot_msg)
  # features.extend([closest_cell.x,
  #                  closest_cell.y,
  #                  0,
  #                  0,
  #                  wall_radius])
  # states = np.array([[features]]) 
  states = np.zeros((1, 1, 34))
  # Generate model input
  # the size should be batch_size x state_dim, where batch_size = 1

  # if update_msg:
  #   h_t = None
  # Not sure if I should update h_t here

  val_preds = []
  val_pred_xs = []
  print(len(models))
  for i, model in enumerate(models):
    # states: seq_len x batch_size x dim
    seq_len = states.shape[0]
    outputs = []
    pred_xs = []
    h_t = h_ts[i]

    for i in range(seq_len):
        cur_states = states[i]

        if i > 0:
          # print("adding new data now ")
          cur_states[:, 0:2] = (new_pred.data).cpu().numpy() # (Variable(x).data).cpu().numpy()
                
        cur_rotated_states = transform_and_rotate(cur_states)
        # now state_t is of size: batch_size x num_human x dim
        batch_size = cur_states.shape[0]
        batch_occupancy_map = []
        
        for b in range(batch_size):
            occupancy_map = build_occupancy_maps(build_humans(cur_states[b]))
            batch_occupancy_map.append(occupancy_map)
        
        batch_occupancy_map = torch.stack(batch_occupancy_map)[:, 1:, :]
        state_t = torch.cat([cur_rotated_states, batch_occupancy_map], dim=-1)
        pred_t, h_t = model(state_t, h_t)
        new_pred = torch.from_numpy(cur_states[:, 0:2]).float() + pred_t[:, 0:2]
        outputs.append(pred_t)
        pred_xs.append(torch.from_numpy(cur_states[:, 0:2]).float() + pred_t[:, 0:2])

    h_ts[i] = h_t
    outputs = torch.stack(outputs)  
          
    pred_xs = torch.stack(pred_xs)

    val_preds.append(outputs)
    val_pred_xs.append(pred_xs)

  val_pred_x, _, val_data_uncertainty, val_model_uncertainty = ensemble(val_preds, val_pred_xs)
    
  print(val_pred_x[0][0][0])
  print(val_pred_x[0][0][1])
  print(val_pred_x.shape)
  print(val_data_uncertainty.shape)
  # output_msg = controller_output()
  # output_msg.dx = val_pred_x[0][0]
  # output_msg.dy = val_pred_x[0][1]
  # output_msg.data_uncertainty = val_data_uncertainty[0].tolist() # Not sure if I could assign list directly
  # output_msg.model_uncertainty = val_model_uncertainty[0].tolist()
  # pub.publish(output_msg)

def main():
  args = parse_args()
  model_config = configparser.RawConfigParser()
  model_config.read(args.model_config)
  if args.bootstrap:
    model_name = args.controller + "_bootstrap"
  else:
    model_name = args.controller + "_ensemble"

  parent_path = os.getcwd()
  checkpoint_file_names = glob(parent_path + "/models/" + model_name + "/model_*.tar")
  for checkpoint_file_name in checkpoint_file_names:
    model = Controller(model_config)
    if os.path.isfile(checkpoint_file_name):
        checkpoint = torch.load(checkpoint_file_name)
        model.load_state_dict(checkpoint['state_dict'])
        print('Loaded checkpoint: %s' % checkpoint_file_name)
        models.append(model)
    else:
        raise ValueError('No such checkpoint: %s' % checkpoint_file_name)
    if args.mc_dropout:
        model.train()
    else:
        model.eval()

  print("Number of loaded models: %d" % (len(models)))



  callback()
  # rospy.init_node(args.controller + '_controler', anonymous=True)
  # pub = rospy.Publisher(args.controller + "_controler_output", controller_output, queue_size=1)

  # sub1 = rospy.Subscriber(PED_TOPIC_NAME, TrackedPersons)
  # sub2 = rospy.Subscriber(OBSTACLE_TOPIC_NAME, GridCells)
  # sub3 = rospy.Subscriber(ROBOT_TOPIC_NAME, Odometry)
  
  # ts = message_filters.TimeSynchronizer([sub1, sub2, sub3], 10)
  # ts.registerCallback(callback)
  # # spin() simply keeps python from exiting until this node is stopped
  # rospy.spin()

if __name__ == '__main__':
  main()