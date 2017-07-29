# -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import tictactoe

with tf.Graph().as_default() as g:
    sess = tf.Session()
    meta_graph = tf.saved_model.loader.load(
        sess=sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        export_dir="model"
    )
    model_signature = meta_graph.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_signature = model_signature.inputs
    output_signature = model_signature.outputs
    # Get names of input and output tensors
    input_tensor_name = input_signature["x"].name
    output_tensor_name = output_signature["y"].name
    # Get input and output tensors
    x_ph = sess.graph.get_tensor_by_name(input_tensor_name)
    y = sess.graph.get_tensor_by_name(output_tensor_name)

env = tictactoe.TicTacToeEnv()
observation = env.reset()
done = False
info = None

rule = """
Input your move!

[0] top-left-square
[1] top-middle-square
[2] top-right-square
[3] middle-left-square
[4] middle-middle-square
[5] middle-right-square
[6] bottom-left-square
[7] bottom-middle-square
[8] bottom-right-square
"""

print(rule)

for _ in range(9):
    env.render()
    if done:
        if info["x"]:
            print("x win!")
        elif info["o"]:
            print("o win!")
        else:
            print("Draw!")
        break
    # Compute scores
    prob_x_win = -np.ones(9)
    prob_o_win = np.ones(9)
    # prob_draw = np.zeros(9)
    for i in range(9):
        if env.board[i] == 0:
            board_copy = np.array([env.board])
            board_copy[0][i] = 1
            prob = sess.run(y, feed_dict={x_ph: board_copy})
            # print i, prob
            prob_x_win[i] = prob[0][0]
            prob_o_win[i] = prob[0][1]
            # prob_draw = prob[0][2]
    # Decide CPU's move
    if max(prob_x_win) >= 0.05:
        cpu_move = prob_x_win.argmax()
    else:
        cpu_move = prob_o_win.argmin()
    _, _, done, info = env.step(cpu_move)
    env.render()
    if done:
        if info["x"]:
            print("x win!")
        elif info["o"]:
            print("o win!")
        else:
            print("Draw!")
        break
    while True:
        sys.stdout.write("Input your move: ")
        player_move = input()
        _, _, done, info = env.step(player_move)
        if info["valid"]:
            break
