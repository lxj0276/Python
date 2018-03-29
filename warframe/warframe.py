# encoding:utf-8
import requests
import xlrd
from xlutils.copy import copy


def find(item):
    h = {
        'authority': 'api.warframe.market',
        'method': 'GET',
        'path': '/v1/items/mirage_prime_chassis/orders',
        'scheme': 'https',
        'accept': 'application/json',
        'accept-language': 'zh-CN,zh;q=0.9',
        'content-type': 'application/json',
        'cookie': '__cfduid=d0b2fa2d66aaf0c7e6b9cf6ff7e4ea7b01522303853; _ga=GA1.2.214173205.1522303868; _gid=GA1.2.1550723962.1522303868; __gads=ID=2d00f43953c2be14:T=1522303883:S=ALNI_MYFzmVapkGLxg6RG7oKVbC0KBNbYg; JWT=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjc3JmX3Rva2VuIjoiZGViODJiNGYzNzlmNmRmZDJlOTVhZGE4YTJjYTQwNGM1N2M5ZTliYyIsImlzcyI6Imp3dCIsInNpZCI6Im8ya3NqT1A1Nnl2b2hrYVdpdW1nc3RmVUxGR1Q2eTdrIiwiYXVkIjoiand0IiwiZXhwIjoxNTI3NDg4ODQ3LCJhdXRoX21ldGhvZCI6ImNvb2tpZSIsImlhdCI6MTUyMjMwNDg0N30.B5AOsIPI4Ywto4p4hP9pPkdknzXjtO_r6BxCAbOKeK8',
        'language': 'en',
        'origin': 'https://warframe.market',
        'platform': 'pc',
        'referer': 'https://warframe.market/items/mirage_prime_chassis',
        'User-Agent': 'Mozilla / 5.0(Windows NT 6.3;Win64;x64) AppleWebKit/537.36(KHTML, likeGecko) Chrome/65.0.3325.162 Safari/537.36'
    }
    url = 'https://api.warframe.market/v1/items/mirage_prime_chassis/orders'
    # 初始化

    h['path'] = '/v1/items/' + item+ '/orders'
    h['referer'] = 'https://warframe.market/items/' + item
    url = 'https://api.warframe.market/v1/items/' + item + '/orders'
    # 修改物件名
    req = requests.get(url, headers=h)
    result = req.json()

    orders = [i for i in result['payload']['orders']]
    pc_orders = list(filter(
        lambda obj: obj['platform'] == 'pc' and obj['order_type'] == 'sell' and obj['user']['status'] == 'ingame',
        orders))

    data = [(i['platinum'], i['user']['ingame_name']) for i in pc_orders]
    data.sort(key=lambda obj: obj[0])
    final_data = data[:5]
    average = sum([i[0] for i in final_data]) / 5
    print(final_data)

    return average


def main():
    file = r"H:\python\warframe\warframe.xls" # 打开指定路径中的xls文件
    book = xlrd.open_workbook(file)  # 得到Excel文件的book对象，实例化对象
    sheet0 = book.sheet_by_index(0)  # 通过sheet索引获得sheet对象
    col = sheet0.col_values(24)

    result = [find(i) for i in col]
    print(result)

    write_book = copy(book)
    write_sheet = write_book.get_sheet(0)
    for i in range(0,len(result)):
        write_sheet.write(i, 25, round(result[i]))
    write_book.save('warframe.xls')
    # 文件操作


if __name__ == '__main__':
    main()
