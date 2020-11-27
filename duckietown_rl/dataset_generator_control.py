#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
from datetime import datetime
import os
from skimage.io import imsave
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--dataset-size', default=50, type=int, help='number of images to generate')
parser.add_argument('--dataset-path', default='datasets/image_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '/', help='location to save the dataset')
parser.add_argument('--compress', action='store_true', help='save the images as a series of png pictures')
parser.add_argument('--split', default=2000, type=int, help='number of images per file (if used without --compress)')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

logging.basicConfig()
logger = logging.getLogger('gym-duckietown')
logger.setLevel(logging.WARNING)

pos_ds = np.zeros((args.dataset_size, env.cur_pos.shape[0]))
angle_ds = np.zeros((args.dataset_size, 1))

if not args.compress:
    image_ds = np.zeros((args.split, env.camera_height, env.camera_width, 3), dtype=np.uint8)
image_count = 0

if not os.path.exists(args.dataset_path):
    os.makedirs(args.dataset_path)

# Generate images
while image_count < args.dataset_size:
    action = np.array([0.0, 0.0])
    obs, _, _, _ = env.step(action)
    pos_ds[image_count] = env.cur_pos
    angle_ds[image_count] = env.cur_angle
    if args.compress:
        imsave(args.dataset_path + str(image_count) + ".png", obs)
    else:
        image_index = image_count % args.split
        if image_index == 0 and image_count > 0:
            file_index = int(image_count / args.split) - 1
            filename = args.dataset_path + "image_ds_" + str(file_index) + ".npy"
            
            print("Saving images in file:", filename)
            np.save(filename, image_ds)
            print("Done.")
        image_ds[image_index] = obs
    env.reset()
    image_count += 1

# Save image dataset
if not args.compress:
    file_index = int(round((image_count - 1) / args.split))
    filename = args.dataset_path + "image_ds_" + str(file_index) + ".npy"
    
    print("Saving images in file:", filename)
    np.save(filename, image_ds[:(image_index + 1)])
    print("Done.")

filename = args.dataset_path + "pos_ds.npy"
    
print("Saving images in file:", filename)
np.save(filename, pos_ds)
print("Done.")

filename = args.dataset_path + "angle_ds.npy"
    
print("Saving images in file:", filename)
np.save(filename, angle_ds)
print("Done.")

env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action += np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, +1])
    if key_handler[key.RIGHT]:
        action += np.array([0, -1])
    
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, _, _, _ = env.step(action)

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
