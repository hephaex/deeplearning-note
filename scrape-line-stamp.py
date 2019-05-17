import requests
import os
from bs4 import BeautifulSoup

def rem(str):
    str0 = str.split("(")[1]
    return str0.split(";")[0]

setname = 0
#chara = "&character=" + str(10)
chara = ""

for page in range(1,11):
    ranking_url = 'https://store.line.me/stickershop/showcase/top_creators/ja?taste=1'+ str(chara) + '&page=' + str(page)
    ran = requests.get(ranking_url)
    soup0 = BeautifulSoup(ran.text, 'lxml')
    stamp_list = soup0.find_all(class_="mdCMN02Li")
    for i in stamp_list:
        target_url = "https://store.line.me" + i.a.get("href")
        r = requests.get(target_url)
        setname += 1
        new_dir_path = str(setname)
        os.makedirs(new_dir_path, exist_ok = True)
        soup = BeautifulSoup(r.text, 'lxml')
        span_list = soup.findAll("span", {"class":"mdCMN09Image"})
        fname = 0
        for i in span_list:
            fname += 1
            imgsrc = rem(i.get("style"))
            req = requests.get(imgsrc)
            if r.status_code == 200:
                f = open( str(setname) + "/" + str(fname) + ".png", 'wb')
                f.write(req.content)
                f.close()
    print("finished downloading page: " + str(page) + " , set: ~" + str(setname) )
