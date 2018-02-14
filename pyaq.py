#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import sys
from board import *
import gtp
import learn
import search


if __name__ == "__main__":
    args = sys.argv

    launch_mode = 0  # 0: gtp, 1: self, 2: learn
    byoyomi = 3.0
    main_time = 0.0
    quick = False
    random = False
    clean = False
    use_gpu = True

    for arg in args:
        if arg.find("self") >= 0:
            launch_mode = 1
        elif arg.find("learn") >= 0:
            launch_mode = 2
        elif arg.find("quick") >= 0:
            quick = True
        elif arg.find("random") >= 0:
            random = True
        elif arg.find("clean") >= 0:
            clean = True
        elif arg.find("main_time") >= 0:
            main_time = float(arg[arg.find("=") + 1:])
        elif arg.find("byoyomi") >= 0:
            byoyomi = float(arg[arg.find("=") + 1:])
        elif arg.find("cpu") >= 0:
            use_gpu = False

    if launch_mode == 0:
        gtp.call_gtp(main_time, byoyomi, quick, clean, use_gpu)

    elif launch_mode == 1:
        b = Board()
        if not random:
            tree = search.Tree("model.ckpt", use_gpu)

        while b.move_cnt < BVCNT * 2:
            prev_move = b.prev_move
            if random:
                move = b.random_play()
            elif quick:
                move = rv2ev(np.argmax(tree.evaluate(b)[0][0]))
                b.play(move, False)
            else:
                move, _ = tree.search(b, 0, clean=clean)
                b.play(move, False)

            b.showboard()
            if prev_move == PASS and move == PASS:
                break

        score_list = []
        b_cpy = Board()

        for i in range(256):
            b.copy(b_cpy)
            b_cpy.rollout(show_board=False)
            score_list.append(b_cpy.score())

        score = Counter(score_list).most_common(1)[0][0]
        if score == 0:
            result_str = "Draw"
        else:
            winner = "B" if score > 0 else "W"
            result_str = "%s+%.1f" % (winner, abs(score))
        sys.stderr.write("result: %s\n" % result_str)

    else:
        learn.learn(3e-4, 0.5, sgf_dir="sgf/", use_gpu=use_gpu, gpu_cnt=1)
