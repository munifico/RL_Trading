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

kospi = []
kosdaq = []
stock_data = pd.read_csv('kospi_stock_code.csv')
kospi = stock_data[['종목코드', '기업명']]
# selenium : 날짜 선택창 조작 위한 라이브러리
driver = webdriver.Chrome("/Users/Eugene/Downloads/chromedriver")
driver.get('http://vip.mk.co.kr/newSt/price/daily.php?stCode='+code)

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
select_m2.select_by_value('02')

select_d2 = Select(driver.find_element_by_name('d2'))
select_d2.select_by_value('28')

# 검색 버튼
btn = driver.find_element_by_xpath("//input[@src='http://img.mk.co.kr/stock/2009/bt_search.gif']")
btn.click()

print("웹 페이지에서 데이터를 로딩중입니다 웹페이지를 종료하지 마십시오")

# 크롤링 전에 5초 쉬면서 로딩 기다림
time.sleep(5)

# 검색된 페이지에서 beautifulSoup 이용해 데이터 가져오기
html = driver.page_source # beautifulsoup한테 검색된 페이지 알려주기
soup = BeautifulSoup(html, 'html5lib')


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
    print(date)
    close = int(td[1].text.replace(",", "")) # td[1]의 text에서 ','를 삭제하고 int로 변환
    open = int(td[4].text.replace(",", ""))
    high = int(td[5].text.replace(",", ""))
    low = int(td[6].text.replace(",", ""))
    value = int(td[7].text.replace(",", ""))
    
    results.append((date, open, close, high, low, value))

table = pd.DataFrame(results, columns=['date', 'open', 'close', 'high', 'low', 'value'])

#print(table)
# db에 저장

con = sqlite3.connect("c:/Users/Eugene/pyquery.db")
cursor = con.cursor()
table.to_sql('stock', con, if_exists='append', index=False)
