from config import STOP_PATH


def stop_load():
    stop_list = [i.strip() for i in open(STOP_PATH, encoding="utf-8").readlines()]
    stop_list.append(" ")
    return stop_list


class TextRankConfig(object):
    TopK = 5
    WINDOW_SIZE = 5
