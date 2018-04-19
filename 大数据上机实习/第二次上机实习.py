# encoding:utf-8
from pymongo import MongoClient
from collections import defaultdict
import redis

def test_dict():
    l = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    d = defaultdict(int)
    for i in l:
        d[i] += 1
    print(sorted(d.keys(), key=lambda obj:d[obj], reverse=True)[0])


def make_dict(names, values):
    return {names[i]: values[i] for i in range(len(names))}


def test_file(n):
    with open('yagoThreeSimplified.txt','r') as f:
        with open('test.txt','w') as fw:
            for i in range(n):
                fw.write(f.readline())


def mdb_store(file, names, db, sep):
    with open(file) as f:
        for line in f:
            line = line.strip('\n')
            db.insert_one(make_dict(names, line.split(sep)))


def mdb_main(file, sep, arg_list, store=False):
    # 连接MongoDB服务
    client = MongoClient('localhost', 27017)
    # 选择数据库及集合
    db = client.test
    lord = db.lord
    names = ['S', 'P', 'O']

    if store:
        mdb_store(file, names, lord, sep)
    # 存入数据库

    # 有S，查所有P，O
    S = arg_list[0]
    rs = lord.find({'S': S})
    for tmp in rs:
        print((tmp['P'], tmp['O']))

    # 有O，查S，P
    O = arg_list[1]
    rs = lord.find({'O': O})
    for tmp in rs:
        print((tmp['P'], tmp['O']))

    # 给出p1，p2，查同时拥有它们的S
    p1, p2 = arg_list[2], arg_list[3]
    rs1 = lord.find({'P': p1})
    rs2 = lord.find({'P': p2})
    result = set([i['S'] for i in rs1]).intersection(set([i['S'] for i in rs2]))
    print(result)

    # 给定O，查拥有O最多的S
    O = arg_list[4]
    rs = lord.find({'O': O})
    d = defaultdict(int)
    for tmp in rs:
        d[tmp['S']] += 1
    print(sorted(d.keys(), key=lambda obj:d[obj], reverse=True)[0] if len(d) > 0 else None)


def redis_store(file, r, sep):
    with open(file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            s, p, o = line.split(' ')
            r.sadd(s, '{} {}'.format(p,o))
            r.sadd(o, '{} {}'.format(s,p))
            r.sadd(p, s)
            r.hsetnx(o+'hash', s, 0) # key-hash表，键值key不能重复
            r.hincrby(o+'hash', s, 1)


def redis_main(file, sep, arg_list, store=False):
    # 连接redis
    r = redis.Redis(host='127.0.0.1', port=6379, db=0)
    if store:
        redis_store(file, r, sep)

    # 有S，查所有P，O
    rs = r.smembers(arg_list[0])
    print('有S，查所有P，O：')
    for i in rs:
        print(i)

    # 有O，查S，P
    rs = r.smembers(arg_list[1])
    print('有O，查S，P：')
    for i in rs:
        print(i)

    # 给出p1，p2，查同时拥有它们的S
    p1, p2 = arg_list[2], arg_list[3]
    rs = r.sinter(p1, p2)
    print('给出p1，p2，查同时拥有它们的S：')
    for i in rs:
        print(i)

    # 给定O，查拥有O最多的S
    rs = r.hgetall(arg_list[4]+'hash')
    print('给定O，查拥有O最多的S：')
    result = sorted(rs.keys(), key=lambda obj:rs[obj], reverse=True)[0] if len(rs)>0 else None
    print(result)
    print(rs.get(result, None))
    print(rs[b'Liam_Tyson'])


def cassandra_main():
    pass


def main():
    S1 = 'Liam_Tyson'
    O1 = 'France'
    P1, P2 = 'isCitizenOf', 'isLeaderOf'
    O2 = 'England'
    arg_list = [S1, O1, P1, P2, O2]
    # mdb_main('yagoThreeSimplified.txt', ' ', arg_list)
    redis_main('test.txt', ' ', arg_list, True)


if __name__ == '__main__':
    main()

