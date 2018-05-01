# encoding:utf-8

from pymongo import MongoClient


def make_dict(names, values):
    return {names[i]: values[i] for i in range(len(names))}


def main():
    with open('student.csv', 'r') as f:
        names = f.readline().split(';')
        data = [make_dict(names, line.split(';')) for line in f.readlines()]

    # 连接MongoDB服务
    client = MongoClient('localhost', 27017)
    # 选择数据库及集合
    db = client.test
    student = db.student
    rs = student.insert_many(data)
    # 插入
    print('Multiple users: {0}'.format(rs.inserted_ids))

    # 检索
    student_tmp = student.find_one({"age": "15"})
    print(student_tmp)

    # 条件查询
    rs = student.find({"Medu": {"$lt": "2"}}).sort("age")
    for tmp in rs:
        print(tmp)

    # 更新文档
    student.update({"id": "mat1"}, {'$set': {"age": 18}})
    # 删除文档
    student.remove({"age": {"$lt": 15}})


if __name__ == '__main__':
    main()