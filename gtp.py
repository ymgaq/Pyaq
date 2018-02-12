#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import stderr, stdout, stdin
from board import *
import numpy as np
from search import Tree


cmd_list = [
    "protocol_version", "name", "version", "list_commands",
    "boardsize", "komi", "time_settings", "time_left",
    "clear_board", "genmove", "play", "undo",
    "gogui-play_sequence", "showboard", "quit"
]


def include(str, cmd):
    return str.find(cmd) >= 0


def send(res_cmd):
    stdout.write("= " + res_cmd + "\n\n")
    stdout.flush()


def args(str):
    arg_list = str.split()
    if arg_list[0] == "=":
        arg_list.pop(0)
    arg_list.pop(0)
    return arg_list


def call_gtp(main_time, byoyomi, quick=False, clean=False, use_gpu=True):
    b = Board()
    tree = Tree(use_gpu=use_gpu)
    tree.main_time = main_time
    tree.byoyomi = byoyomi

    while 1:
        str = stdin.readline().rstrip("\r\n")
        if str == "":
            continue
        elif include(str, "protocol_version"):
            send("2")
        elif include(str, "name"):
            send("Pyaq")
        elif include(str, "version"):
            send("1.0")
        elif include(str, "list_commands"):
            stdout.write("=")
            for cmd in cmd_list:
                stdout.write(cmd + "\n")
            send("")
        elif include(str, "boardsize"):
            bs = int(args(str)[0])
            if bs != BSIZE:
                stdout.write("?invalid boardsize\n\n")
            send("")
        elif include(str, "komi"):
            send("")
        elif include(str, "time_settings"):
            arg_list = args(str)
            tree.main_time = arg_list[0]
            tree.left_time = tree.main_time
            tree.byoyomi = arg_list[1]
        elif include(str, "time_left"):
            tree.left_time = float(args(str)[1])
        elif include(str, "clear_board"):
            b.clear()
            tree.clear()
            send("")
        elif include(str, "genmove"):
            if quick:
                move = rv2ev(np.argmax(tree.evaluate(b)[0][0]))
            else:
                move, win_rate = tree.search(b, 0, ponder=False, clean=clean)

            if win_rate < 0.1:
                send("resign")
            else:
                b.play(move)
                send(ev2str(move))
        elif include(str, "play"):
            b.play(str2ev(args(str)[1]), not_fill_eye=False)
            send("")
        elif include(str, "undo"):
            history = b.history[:-1]
            b.clear()
            tree.clear()
            for v in history:
                b.play(v, not_fill_eye=False)

            send("")
        elif include(str, "gogui-play_sequence"):
            arg_list = args(str)
            for i in range(1, len(arg_list) + 1, 2):
                b.play(str2ev(arg_list[i]), not_fill_eye=False)

            send("")
        elif include(str, "showboard"):
            b.showboard()
            send("")
        elif include(str, "quit"):
            send("")
            break
        else:
            stdout.write("?unknown_command\n\n")
