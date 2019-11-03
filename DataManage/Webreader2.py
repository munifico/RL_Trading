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

class Webreader2:
    def __init__(self):
        self.driver = self.create_selenium_driver()


    def create_selenium_driver(self):
        driver = webdriver.Chrome("/Users/Eugene/Downloads/chromedriver")
        driver.get('http://vip.mk.co.kr/newSt/price/daily.php?stCode=005930')
        return driver

    def crawler(self, y1, m1, d1, y2, m2, d2):
        print("실행중")
        select_y1 = Select(self.driver.find_element_by_name('y1'))
        select_y1.select_by_value(y1)

        select_m1 = Select(self.driver.find_element_by_name('m1'))
        select_m1.select_by_value(m1)

        select_d1 = Select(self.driver.find_element_by_name('d1'))
        select_d1.select_by_value(d1)

        select_y2 = Select(self.driver.find_element_by_name('y2'))
        select_y2.select_by_value(y2)

        select_m2 = Select(self.driver.find_element_by_name('m2'))
        select_m2.select_by_value(m2)

        select_d2 = Select(self.driver.find_element_by_name('d2'))
        select_d2.select_by_value(d2)
        btn = self.driver.find_element_by_xpath("//input[@src='http://img.mk.co.kr/stock/2009/bt_search.gif']")
        btn.click()

        # 크롤링 전에 5초 쉬면서 로딩 기다림
        time.sleep(5)

        html = self.driver.page_source
        soup = BeautifulSoup(html, 'html5lib')

        table = soup.find('table', attrs={'class': 'table_3'})
        trs = table.findAll('tr')  # 테이블의 모든 row들 저장

        results = []  # row에 있는 데이터를 저장

        for row in trs:
            if row == trs[0]:
                continue
            td = row.findAll('td')
            date = td[0].text
            year = int(date[0:2])
            if year == 99:
                year = str(1900 + year)
            else:
                year = str(2000 + year)
            month = date[3:5]
            day = date[6:8]
            date = year + "-" + month + "-" + day
            #date = td[0].text
            close = int(td[1].text.replace(",", ""))  # td[1]의 text에서 ','를 삭제하고 int로 변환
            open = int(td[4].text.replace(",", ""))
            high = int(td[5].text.replace(",", ""))
            low = int(td[6].text.replace(",", ""))
            value = int(td[7].text.replace(",", ""))

            results.append((date, open, high, low, close, value))

            table = pd.DataFrame(results, columns=['date', 'open', 'high', 'low', 'close', 'value'])
        print("완료")
        return table
'''
if __name__ == "__main__":
    webreader = Webreader2()
    webreader.create_selenium_driver()
'''