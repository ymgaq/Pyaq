# -*- coding: utf-8 -*-

from math import sqrt
from sys import stderr
import time
from board import *
import model
import numpy as np
import tensorflow as tf

max_node_cnt = 2 ** 14  # 16384
expand_cnt = 8


class Node(object):
    def __init__(self):
        self.init_branch()
        self.clear()

    def clear(self):
        self.branch_cnt = 0
        self.total_value = 0.0
        self.total_cnt = 0
        self.hash = 0
        self.move_cnt = -1

    def init_branch(self):
        self.move = np.full(BVCNT + 1, VNULL)
        self.prob = np.full(BVCNT + 1, 0.0)
        self.value = np.full(BVCNT + 1, 0.0)
        self.value_win = np.full(BVCNT + 1, 0.0)
        self.visit_cnt = np.full(BVCNT + 1, 0)
        self.next_id = np.full(BVCNT + 1, -1)
        self.next_hash = np.full(BVCNT + 1, -1)
        self.evaluated = np.full(BVCNT + 1, False)


class Tree(object):
    cp = 2.0
    stop = False

    def __init__(self, ckpt_path="model.ckpt", use_gpu=True):
        self.set_sess(ckpt_path, use_gpu)
        self.node = [Node() for _ in range(max_node_cnt)]
        self.main_time = 0.0
        self.byoyomi = 1.0
        self.clear()

    def clear(self):
        self.left_time = self.main_time
        for nd in self.node:
            nd.clear()
        self.node_cnt = 0
        self.root_id = 0
        self.root_move_cnt = 0
        self.node_hashs = {}
        self.eval_cnt = 0
        Tree.stop = False

    def set_sess(self, ckpt_path, use_gpu=True):
        device_name = "gpu" if use_gpu else "cpu"
        with tf.get_default_graph().as_default(), tf.device("/%s:0" % device_name):
            dn = model.DualNetwork()
            self.x = tf.placeholder("float", shape=[None, BVCNT, FEATURE_CNT])
            self.pv = dn.model(self.x, temp=0.7, dr=1.0)
            self.sess = dn.create_sess(ckpt_path)

    def evaluate(self, b):
        return self.sess.run(self.pv,
                             feed_dict={self.x: np.reshape(b.feature(), (1, BVCNT, 7))})

    def delete_node(self):
        if self.node_cnt < max_node_cnt * 0.5:
            return
        for i in range(max_node_cnt):
            mc = self.node[i].move_cnt
            if 0 <= mc and mc < self.root_move_cnt:
                if self.node[i].hash in self.node_hashs:
                    self.node_hashs.pop(self.node[i].hash)
                self.node[i].clear()

    def create_node(self, b_info, prob):
        hs = b_info[0]

        if hs in self.node_hashs and \
                self.node[self.node_hashs[hs]].hash == hs and \
                self.node[self.node_hashs[hs]].move_cnt == b_info[1]:
            return self.node_hashs[hs]

        node_id = hs % max_node_cnt
        while self.node[node_id].move_cnt != -1:
            node_id = node_id + 1 if node_id + 1 < max_node_cnt else 0
        self.node_hashs[hs] = node_id
        self.node_cnt += 1

        nd = self.node[node_id]
        nd.clear()
        nd.move_cnt = b_info[1]
        nd.hash = hs
        nd.init_branch()

        order_ = np.argsort(prob)[::-1]
        for rv in order_:
            if rv in b_info[2]:
                nd.move[nd.branch_cnt] = rv2ev(rv)
                nd.prob[nd.branch_cnt] = prob[rv]
                nd.branch_cnt += 1

        return node_id

    def search_branch(self, b, node_id, route):
        nd = self.node[node_id]

        nd_rate = 0.0 if nd.total_cnt == 0 else nd.total_value / nd.total_cnt
        cpsv = Tree.cp * sqrt(nd.total_cnt)

        with np.errstate(divide='ignore', invalid='ignore'):
            rate = nd.value_win / nd.visit_cnt  # including dividing by 0
            rate[~np.isfinite(rate)] = nd_rate  # convert nan, inf to nd_rate
        action_value = rate + cpsv * nd.prob / (nd.visit_cnt + 1)
        best = np.argmax(action_value[:nd.branch_cnt])

        route.append((node_id, best))
        next_id = nd.next_id[best]
        next_move = nd.move[best]

        head_node = not self.has_next(node_id, best, b.move_cnt + 1) or \
            nd.visit_cnt[best] < expand_cnt or \
            (b.move_cnt > BVCNT * 2) or \
            (next_move == PASS and b.prev_move == PASS)

        b.play(next_move, False)

        if head_node:
            if nd.evaluated[best]:
                value = nd.value[best]
            else:
                prob_, value_ = self.evaluate(b)
                self.eval_cnt += 1
                value = -value_[0]
                nd.value[best] = value
                nd.evaluated[best] = True

                if self.node_cnt > 0.85 * max_node_cnt:
                    self.delete_node()

                next_id = self.create_node(b.info(), prob_[0])
                next_nd = self.node[next_id]
                nd.next_id[best] = next_id
                nd.next_hash[best] = b.hash()

                next_nd.total_value -= nd.value_win[best]
                next_nd.total_cnt += nd.visit_cnt[best]
        else:
            value = -self.search_branch(b, next_id, route)

        nd.total_value += value
        nd.total_cnt += 1
        nd.value_win[best] += value
        nd.visit_cnt[best] += 1

        return value

    def search(self, b, time_, ponder=False, clean=False):
        start = time.time()

        prob, _ = self.evaluate(b)
        self.root_id = self.create_node(b.info(), prob[0])
        self.root_move_cnt = b.move_cnt
        Tree.cp = 0.01 if b.move_cnt < 8 else 1.5

        nd = self.node[self.root_id]

        if nd.branch_cnt <= 1:
            stderr.write("\nmove count=%d:\n" % (b.move_cnt + 1))
            self.print_info(self.root_id)
            return PASS, 0.5

        self.delete_node()
        order_ = np.argsort(nd.visit_cnt[:nd.branch_cnt])[::-1]
        best, second = tuple(order_[:2].tolist())

        win_rate = self.branch_rate(nd, best)

#         if not ponder and self.byoyomi == 0 and self.left_time < 10:
#             if nd.visit_cnt[best] < 1000:
#                 return rv2ev(np.argmax(prob)), 0.5
#             else:
#                 stderr.write("\nmove count=%d:\n" % (b.move_cnt + 1))
#                 self.print_info(self.root_id)
#                 return nd.move[best], win_rate

        stand_out = nd.total_cnt > 5000 and nd.visit_cnt[best] > nd.visit_cnt[second] * 100
        almost_win = nd.total_cnt > 5000 and (win_rate < 0.1 or win_rate > 0.9)

        if ponder or not (stand_out or almost_win):
            if time_ == 0:
                if self.main_time == 0 or self.left_time < self.byoyomi * 2:
                    time_ = max(self.byoyomi, 1.0)
                else:
                    time_ = self.left_time / (55.0 + max(50 - b.move_cnt, 0.0))

            # search
            search_idx = 1
            self.eval_cnt = 0
            b_cpy = Board()
            while 1:
                b.copy(b_cpy)
                route = []

                self.search_branch(b_cpy, self.root_id, route)
                search_idx += 1
                if search_idx % 64 == 0:
                    if (ponder and Tree.stop) or time.time() - start > time_:
                        Tree.stop = False
                        break

            order_ = np.argsort(nd.visit_cnt[:nd.branch_cnt])[::-1]
            best, second = tuple(order_[:2].tolist())

        next_move = nd.move[best]
        win_rate = self.branch_rate(nd, best)

        if clean and next_move == PASS:
            if nd.value_win[best] * nd.value_win[second] > 0:
                next_move = nd.move[second]
                win_rate = self.branch_rate(nd, second)

        if not ponder:
            stderr.write("\nmove count=%d: left time=%.1f[sec] evaluated=%d\n" % (
                b.move_cnt + 1, max(self.left_time - time_, 0), self.eval_cnt))
            self.print_info(self.root_id)
            self.left_time = max(0.0, self.left_time - (time.time() - start))

        return next_move, win_rate

    def has_next(self, node_id, br_id, move_cnt):
        nd = self.node[node_id]
        next_id = nd.next_id[br_id]
        return next_id >= 0 and \
            nd.next_hash[br_id] == self.node[next_id].hash and \
            self.node[next_id].move_cnt == move_cnt

    def branch_rate(self, nd, id):
        return nd.value_win[id] / max(nd.visit_cnt[id], 1) / 2 + 0.5

    def best_sequence(self, node_id, head_move):
        seq_str = "%-3s" % ev2str(head_move)
        next_move = head_move

        for i in range(7):
            nd = self.node[node_id]
            if next_move == PASS or nd.branch_cnt < 1:
                break

            best = np.argmax(nd.visit_cnt[:nd.branch_cnt])
            if nd.visit_cnt[best] == 0:
                break
            next_move = nd.move[best]
            seq_str += "->%-3s" % ev2str(next_move)

            if not self.has_next(node_id, best, nd.move_cnt + 1):
                break
            node_id = nd.next_id[best]

        return seq_str

    def print_info(self, node_id):
        nd = self.node[node_id]
        order_ = np.argsort(nd.visit_cnt[:nd.branch_cnt])[::-1]
        stderr.write("|move|count  |rate |value|prob | best sequence\n")

        for i in range(min(len(order_), 9)):
            m = order_[i]
            visit_cnt = nd.visit_cnt[m]
            if visit_cnt == 0:
                break

            rate = 0.0 if visit_cnt == 0 else self.branch_rate(nd, m) * 100
            value = (nd.value[m] / 2 + 0.5) * 100

            stderr.write("|%-4s|%7d|%5.1f|%5.1f|%5.1f| %s\n" % (
                ev2str(nd.move[m]), visit_cnt, rate, value, nd.prob[m] * 100,
                self.best_sequence(nd.next_id[m], nd.move[m])))
