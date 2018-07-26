import os
from config import CLASSIFY_RULES, TARGET_DIR


def classify(rules, target):
    os.chdir(target)
    for i in rules.keys():
        try:
            os.mkdir(i)
        except FileExistsError:
            pass

    invert_rule = {i: j for j in rules.keys() for i in rules[j]}
    for root, _, files in os.walk('./'):
        for file in files:
            fmt = file.split('.')[-1]
            try:
                os.rename('./{}'.format(file), './{}/{}'.format(invert_rule[fmt], file))
            except FileNotFoundError:
                pass


def test():
    os.rename('test/1.txt', 'ok.txt')


if __name__ == '__main__':
    classify(CLASSIFY_RULES, TARGET_DIR)