import os
import re
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
import pandas as pd
import sqlite3
from datetime import datetime
import Project.codereader as cr

os.chdir("/Users/unanimous0/RL_CompetitiveTradingModel_ver0_1/Project")

cnt = 0
# 디비 경로부분은 컴퓨터 상황에 맞게 바꿔주세요! Project폴더의 kosdaq.db의 경로로
con = sqlite3.connect("kosdaq.db")
cursor = con.cursor()

def create_table(name):
    sql = '''
        CREATE TABLE IF NOT EXISTS %s (
            Date text,
            Open int,
            High int,
            Low int,
            Close int,
            Volume int
        );
    ''' % name
    cursor.execute(sql)
    con.commit()



def Webreader(code):
    # selenium : 날짜 선택창 조작 위한 라이브러리
    # !!! 드라이버를 https://sites.google.com/a/chromium.org/chromedriver/downloads 여기서 받아서
    # !!! 밑부분의 경로를 알맞게 바꿔주세요!
    driver = webdriver.Chrome("/Users/Eugene/Downloads/chromedriver")
    driver.get('http://vip.mk.co.kr/newSt/price/daily.php?stCode='+code)
    create_table('code')
    # 시작 년도 설정 : 나중에 마지막 갱신일을 기준으로 다시 설정해야함
    select_y1 = Select(driver.find_element_by_name('y1'))
    select_y1.select_by_value('1999')

    select_m1 = Select(driver.find_element_by_name('m1'))
    select_m1.select_by_value('01')

    select_d1 = Select(driver.find_element_by_name('d1'))
    select_d1.select_by_value('01')

    # 종료 년도 설정 : 나중에 현재 날짜를 받아서 하는 걸로 변경해야 함
    select_y2 = Select(driver.find_element_by_name('y2'))
    select_y2.select_by_value('2019')

    select_m2 = Select(driver.find_element_by_name('m2'))
    select_m2.select_by_value('03')

    select_d2 = Select(driver.find_element_by_name('d2'))
    select_d2.select_by_value('02')

    # 검색 버튼
    btn = driver.find_element_by_xpath("//input[@src='http://img.mk.co.kr/stock/2009/bt_search.gif']")
    btn.click()

    print("웹 페이지에서 데이터를 로딩중입니다 웹페이지를 종료하지 마십시오")

    while True:
        # 검색된 페이지에서 beautifulSoup 이용해 데이터 가져오기
        html = driver.page_source  # beautifulsoup한테 검색된 페이지 알려주기
        soup = BeautifulSoup(html, 'html5lib')
        next_button = soup.find('img', alt='다음')
        if next_button==None:
            driver.quit()
            table = 0
            return table
        next = driver.find_element_by_xpath("//img[@src='http://img.mk.co.kr/stock/2009/bt_next.gif']")
        next.click()
        # 크롤링 전에 5초 쉬면서 로딩 기다림
        time.sleep(5)



        dates = soup.findAll('td', attrs={'class':'center'})
        table = soup.find('table', attrs={'class':'table_3'})
        trs = table.findAll('tr') #테이블의 모든 row들 저장


        results = [] #row에 있는 데이터를 저장

        # 테이블의 모든 row에 대해 한 줄마다 데이터를 가져옴
        for row in trs:
            if row==trs[0]:
                continue
            td = row.findAll('td')
            date = td[0].text
            year = int(date[0:2])
            if year==99:
                year = str(1900+year)
            else:
                year = str(2000+year)
            month = date[3:5]
            day = date[6:8]
            date = year+"-"+month+"-"+day
            close = int(td[1].text.replace(",", "")) # td[1]의 text에서 ','를 삭제하고 int로 변환
            open = int(td[4].text.replace(",", ""))
            high = int(td[5].text.replace(",", ""))
            low = int(td[6].text.replace(",", ""))
            volume = int(td[7].text.replace(",", ""))

            results.append((date, open, high, low, close, volume))

        table = pd.DataFrame(results, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        table.to_sql(code, con, if_exists='append', index=False)









if __name__ == "__main__":
    kosdaq_codes = cr.read_kosdaq_code()
    kospi_codes = cr.read_kospi_code()

    for code in kosdaq_codes:
        table = Webreader(code)
        if table==0:
            cnt = cnt+1
            print(cnt)
            continue




'''
con = sqlite3.connect("c:/Users/Eugene/pyquery.db")
cursor = con.cursor()
table.to_sql('stock', con, if_exists='append', index=False)
'''