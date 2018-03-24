#encoding:utf-8
from bs4 import BeautifulSoup
import requests

req = requests.get('http://www.carnoc.com/mhzl/jchzl/airport3code.htm')
soup = BeautifulSoup(req.text, 'html.parser')

with open('cityCode.txt','w') as f:
    for tr in soup.find('tbody').select('tr')[1:]:
        tdList = list(tr.select('td'))
        for td1, td2 in zip(tdList[::2], tdList[1::2]):
            f.write(td1.get_text()+' '+td2.get_text()+'\n')
