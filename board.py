# -*- coding: utf-8 -*-


from sys import stderr
import numpy as np


BSIZE = 9  # board size
EBSIZE = BSIZE + 2  # extended board size
BVCNT = BSIZE ** 2  # vertex count
EBVCNT = EBSIZE ** 2  # extended vertex count
PASS = EBVCNT  # pass
VNULL = EBVCNT + 1  # invalid position
KOMI = 7.0
dir4 = [1, EBSIZE, -1, -EBSIZE]
diag4 = [1 + EBSIZE, EBSIZE - 1, -EBSIZE - 1, 1 - EBSIZE]
KEEP_PREV_CNT = 2
FEATURE_CNT = KEEP_PREV_CNT * 2 + 3  # 7
x_labels = "ABCDEFGHJKLMNOPQRST"


def ev2xy(ev):
    return ev % EBSIZE, ev // EBSIZE


def xy2ev(x, y):
    return y * EBSIZE + x


def rv2ev(rv):
    if rv == BVCNT:
        return PASS
    return rv % BSIZE + 1 + (rv // BSIZE + 1) * EBSIZE


def ev2rv(ev):
    if ev == PASS:
        return BVCNT
    return ev % EBSIZE - 1 + (ev // EBSIZE - 1) * BSIZE


def ev2str(ev):
    if ev >= PASS:
        return "pass"
    x, y = ev2xy(ev)
    return x_labels[x - 1] + str(y)


def str2ev(v_str):
    v_str = v_str.upper()
    if v_str == "PASS" or v_str == "RESIGN":
        return PASS
    else:
        x = x_labels.find(v_str[0]) + 1
        y = int(v_str[1:])
        return xy2ev(x, y)


rv_list = [rv2ev(i) for i in range(BVCNT)]


class StoneGroup(object):

    def __init__(self):
        self.lib_cnt = VNULL  # liberty count
        self.size = VNULL  # stone size
        self.v_atr = VNULL  # liberty position if in Atari
        self.libs = set()  # set of liberty positions

    def clear(self, stone=True):
        # clear as placed stone or empty
        self.lib_cnt = 0 if stone else VNULL
        self.size = 1 if stone else VNULL
        self.v_atr = VNULL
        self.libs.clear()

    def add(self, v):
        # add liberty at v
        if v not in self.libs:
            self.libs.add(v)
            self.lib_cnt += 1
            self.v_atr = v

    def sub(self, v):
        # remove liberty at v
        if v in self.libs:
            self.libs.remove(v)
            self.lib_cnt -= 1

    def merge(self, other):
        # merge with aother stone group
        self.libs |= other.libs
        self.lib_cnt = len(self.libs)
        self.size += other.size
        if self.lib_cnt == 1:
            for lib in self.libs:
                self.v_atr = lib


class Board(object):

    def __init__(self):
        # 1-d array ([EBVCNT]) of stones or empty or exterior
        # 0: white 1: black
        # 2: empty 3: exterior
        self.color = np.full(EBVCNT, 3)
        self.sg = [StoneGroup() for _ in range(EBVCNT)]  # stone groups
        self.clear()

    def clear(self):
        self.color[rv_list] = 2  # empty
        self.id = np.arange(EBVCNT)  # id of stone group
        self.next = np.arange(EBVCNT)  # next position in the same group
        for i in range(EBVCNT):
            self.sg[i].clear(stone=False)
        self.prev_color = [np.copy(self.color) for _ in range(KEEP_PREV_CNT)]

        self.ko = VNULL  # illegal position due to Ko
        self.turn = 1  # black
        self.move_cnt = 0  # move count
        self.prev_move = VNULL  # previous move
        self.remove_cnt = 0  # removed stones count
        self.history = []

    def copy(self, b_cpy):
        b_cpy.color = np.copy(self.color)
        b_cpy.id = np.copy(self.id)
        b_cpy.next = np.copy(self.next)
        for i in range(EBVCNT):
            b_cpy.sg[i].lib_cnt = self.sg[i].lib_cnt
            b_cpy.sg[i].size = self.sg[i].size
            b_cpy.sg[i].v_atr = self.sg[i].v_atr
            b_cpy.sg[i].libs |= self.sg[i].libs
        for i in range(KEEP_PREV_CNT):
            b_cpy.prev_color[i] = np.copy(self.prev_color[i])

        b_cpy.ko = self.ko
        b_cpy.turn = self.turn
        b_cpy.move_cnt = self.move_cnt
        b_cpy.prev_move = self.prev_move
        b_cpy.remove_cnt = self.remove_cnt

        for h in self.history:
            b_cpy.history.append(h)

    def remove(self, v):
        # remove stone group including stone at v
        v_tmp = v
        while 1:
            self.remove_cnt += 1
            self.color[v_tmp] = 2  # empty
            self.id[v_tmp] = v_tmp  # reset id
            for d in dir4:
                nv = v_tmp + d
                # add liberty to neighbor groups
                self.sg[self.id[nv]].add(v_tmp)
            v_next = self.next[v_tmp]
            self.next[v_tmp] = v_tmp
            v_tmp = v_next
            if v_tmp == v:
                break  # finish when all stones are removed

    def merge(self, v1, v2):
        # merge stone groups at v1 and v2
        id_base = self.id[v1]
        id_add = self.id[v2]
        if self.sg[id_base].size < self.sg[id_add].size:
            id_base, id_add = id_add, id_base  # swap
        self.sg[id_base].merge(self.sg[id_add])

        v_tmp = id_add
        while 1:
            self.id[v_tmp] = id_base  # change id to id_base
            v_tmp = self.next[v_tmp]
            if v_tmp == id_add:
                break
        # swap next id for circulation
        self.next[v1], self.next[v2] = self.next[v2], self.next[v1]

    def place_stone(self, v):
        self.color[v] = self.turn
        self.id[v] = v
        self.sg[self.id[v]].clear(stone=True)
        for d in dir4:
            nv = v + d
            if self.color[nv] == 2:
                self.sg[self.id[v]].add(nv)  # add liberty
            else:
                self.sg[self.id[nv]].sub(v)  # remove liberty
        # merge stone groups
        for d in dir4:
            nv = v + d
            if self.color[nv] == self.turn and self.id[nv] != self.id[v]:
                self.merge(v, nv)
        # remove opponent's stones
        self.remove_cnt = 0
        for d in dir4:
            nv = v + d
            if self.color[nv] == int(self.turn == 0) and \
                    self.sg[self.id[nv]].lib_cnt == 0:
                self.remove(nv)

    def legal(self, v):
        if v == PASS:
            return True
        elif v == self.ko or self.color[v] != 2:
            return False

        stone_cnt = [0, 0]
        atr_cnt = [0, 0]
        for d in dir4:
            nv = v + d
            c = self.color[nv]
            if c == 2:
                return True
            elif c <= 1:
                stone_cnt[c] += 1
                if self.sg[self.id[nv]].lib_cnt == 1:
                    atr_cnt[c] += 1

        return (atr_cnt[int(self.turn == 0)] != 0 or
                atr_cnt[self.turn] < stone_cnt[self.turn])

    def eyeshape(self, v, pl):
        if v == PASS:
            return False
        for d in dir4:
            c = self.color[v + d]
            if c == 2 or c == int(pl == 0):
                return False

        diag_cnt = [0, 0, 0, 0]
        for d in diag4:
            nv = v + d
            diag_cnt[self.color[nv]] += 1

        wedge_cnt = diag_cnt[int(pl == 0)] + int(diag_cnt[3] > 0)
        if wedge_cnt == 2:
            for d in diag4:
                nv = v + d
                if self.color[nv] == int(pl == 0) and \
                        self.sg[self.id[nv]].lib_cnt == 1 and \
                        self.sg[self.id[nv]].v_atr != self.ko:
                    return True

        return wedge_cnt < 2

    def play(self, v, not_fill_eye=True):

        if not self.legal(v):
            return 1
        elif not_fill_eye and self.eyeshape(v, self.turn):
            return 2
        else:
            for i in range(KEEP_PREV_CNT - 1)[::-1]:
                self.prev_color[i + 1] = np.copy(self.prev_color[i])
            self.prev_color[0] = np.copy(self.color)

            if v == PASS:
                self.ko = VNULL
            else:
                self.place_stone(v)
                id = self.id[v]
                self.ko = VNULL
                if self.remove_cnt == 1 and \
                        self.sg[id].lib_cnt == 1 and \
                        self.sg[id].size == 1:
                    self.ko = self.sg[id].v_atr

        self.prev_move = v
        self.history.append(v)
        self.turn = int(self.turn == 0)
        self.move_cnt += 1

        return 0

    def random_play(self):
        empty_list = np.where(self.color == 2)[0]
        np.random.shuffle(empty_list)

        for v in empty_list:
            if self.play(v, True) == 0:
                return v

        self.play(PASS)
        return PASS

    def score(self):
        stone_cnt = [0, 0]
        for rv in range(BVCNT):
            v = rv2ev(rv)
            c = self.color[v]
            if c <= 1:
                stone_cnt[c] += 1
            else:
                nbr_cnt = [0, 0, 0, 0]
                for d in dir4:
                    nbr_cnt[self.color[v + d]] += 1
                if nbr_cnt[0] > 0 and nbr_cnt[1] == 0:
                    stone_cnt[0] += 1
                elif nbr_cnt[1] > 0 and nbr_cnt[0] == 0:
                    stone_cnt[1] += 1
        return stone_cnt[1] - stone_cnt[0] - KOMI

    def rollout(self, show_board=False):
        while self.move_cnt < EBVCNT * 2:
            prev_move = self.prev_move
            move = self.random_play()
            if show_board and move != PASS:
                stderr.write("\nmove count=%d\n" % self.move_cnt)
                self.showboard()
            if prev_move == PASS and move == PASS:
                break

    def showboard(self):

        def print_xlabel():
            line_str = "  "
            for x in range(BSIZE):
                line_str += " " + x_labels[x] + " "
            stderr.write(line_str + "\n")

        print_xlabel()

        for y in range(1, BSIZE + 1)[::-1]:  # 9, 8, ..., 1
            line_str = str(y) if y >= 10 else " " + str(y)
            for x in range(1, BSIZE + 1):
                v = xy2ev(x, y)
                x_str = " . "
                color = self.color[v]
                if color <= 1:
                    stone_str = "O" if color == 0 else "X"
                    if v == self.prev_move:
                        x_str = "[" + stone_str + "]"
                    else:
                        x_str = " " + stone_str + " "
                line_str += x_str
            line_str += str(y) if y >= 10 else " " + str(y)
            stderr.write(line_str + "\n")

        print_xlabel()
        stderr.write("\n")

    def feature(self):
        feature_ = np.zeros((EBVCNT, FEATURE_CNT), dtype=np.float)
        my = self.turn
        opp = int(self.turn == 0)

        feature_[:, 0] = (self.color == my)
        feature_[:, 1] = (self.color == opp)
        for i in range(KEEP_PREV_CNT):
            feature_[:, (i + 1) * 2] = (self.prev_color[i] == my)
            feature_[:, (i + 1) * 2 + 1] = (self.prev_color[i] == opp)
        feature_[:, FEATURE_CNT - 1] = my

        return feature_[rv_list, :]

    def hash(self):
        return (hash(self.color.tostring()) ^
                hash(self.prev_color[0].tostring()) ^ self.turn)

    def info(self):
        empty_list = np.where(self.color == 2)[0]
        cand_list = []
        for v in empty_list:
            if self.legal(v) and not self.eyeshape(v, self.turn):
                cand_list.append(ev2rv(v))
        cand_list.append(ev2rv(PASS))
        return (self.hash(), self.move_cnt, cand_list)
