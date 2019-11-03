import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from Project.Webreader2 import Webreader2
from Project.Kiwoom import Kiwoom
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from pandas import DataFrame


MARKET_KOSPI = 0
MARKET_KOSDAQ = 10

stock_column_idx_lookup = {0:'date', 1:'open', 2:'close', 3:'high', 4:'low'}
form_class = uic.loadUiType("pyquery.ui")[0]




class Mywindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        # ui 셋업
        self.setupUi(self)
        #db 연결
        self.con = sqlite3.connect("c:/Users/Eugene/PycharmProjects/RL_CompetitiveTradingModel_ver0_1/rl_trader.db")
        self.cursor = self.con.cursor()
        '''
        # 키움 연결
        self.kiwoom = Kiwoom()
        self.kiwoom.comm_connect()

        self.get_code_list()
        self.stock_sdateEdit.textChanged.connect(self.code_changed)
        self.load_stock_data()
        '''

        # 갱신버튼
        self.renewalBtn.clicked.connect(self.renewal_btn_clicked)
        # 주식 데이터 조회 버튼
        self.stock_btn.clicked.connect(self.stock_btn_clicked)



    # 종료날짜 가져오기 : 나중에 예외 처리해주기(테이블에 데이터가 없을 경우에 대비해서)
    def get_lastdate(self):
        lastdate = []
        con2 = sqlite3.connect("c:/Users/Eugene/PycharmProjects/RL_CompetitiveTradingModel_ver0_1/rl_trader.db")
        cursor2 = con2.cursor()
        cursor2.execute("SELECT year FROM update order by year desc, month desc, day desc LIMIT 1")
        lastdate.append(cursor2.fetchone())
        cursor2.execute("SELECT month FROM update order by year desc, month desc, day desc LIMIT 1")
        lastdate.append(cursor2.fetchone())
        cursor2.execute("SELECT day FROM update order by year desc, month desc, day desc LIMIT 1")
        lastdate.append(cursor2.fetchone())
        # 가져온 년,월,일을 리스ㅡ에 저장하여 반환
        n_lastdate = []
        i=0
        while i<3:
            n_lastdate.append(lastdate[i][0])
            i = i+1
        return n_lastdate

    # 현재날짜 가져오기
    def get_currentdate(self):
        now = datetime.now()
        cyear = now.strftime('%Y')
        cmonth = now.strftime('%m')
        cday = now.strftime('%d')
        currentdate = [cyear, cmonth, cday]
        return currentdate

    # 마지막 갱신일의 다음날 찾기
    def date_to_update(self, original):
        year = int(original[0])
        month = int(original[1])
        day = int(original[2])
        last_date = datetime(year, month, day)
        new_date = last_date + timedelta(days=1)
        new_year = new_date.strftime('%Y')
        new_month = new_date.strftime('%m')
        new_day = new_date.strftime('%d')
        target=[new_year, new_month, new_day]
        return target

    # 갱신 버튼 이벤트
    def renewal_btn_clicked(self):
        # 디비에서 마지막 갱신날짜 가져오기
        ldate=self.get_lastdate()
        # 마지막 갱신일의 다음날 찾기
        new_ldate = self.date_to_update(ldate)
        # 현재 날짜 저장
        cdate = self.get_currentdate()
        # webreader모듈에서 crawler사용
        webreader = Webreader2()
        stock_data = webreader.crawler(new_ldate[0], new_ldate[1], new_ldate[2], cdate[0], cdate[1], cdate[2])
        # 데이터 저장 : 저장할 때 년도 포맷 바꿔줘야 함(네자리로)
        stock_data.to_sql('SamsungElectronics', self.con, if_exists='append', index=False)
        # 현재날짜를 디비에 저장(마지막 갱신일로 저장하는 것임)
        cursor3 = self.con.cursor()
        sql = "insert into update (year, month, day) values (?, ?, ?)"
        cursor3.execute(sql, cdate)
        self.con.commit()
        cursor3.close()

        # 새로운 갱신일로 레이블 바꿔주기
        new_ldate = self.get_lastdate()
        self.lastUpdate.setText(new_ldate[0]+"/"+new_ldate[1]+"/"+new_ldate[2])


    '''
    def get_code_list(self):
        self.kospi_codes = self.kiwoom.get_code_list_by_market(MARKET_KOSPI)
        self.kosdaq_codes = self.kiwoom.get_code_list_by_market(MARKET_KOSDAQ)

    def code_changed(self):
        code = self.stock_codeEdit.text()
        name = self.kiwoom.get_master_code_name(code)
        self.stock_codeEdit.setText(name)
    '''
    '''
    def stock_sdate_changed(self):

    
    def setTableWidgetData(self, table, code, sdate, edate, type, sql):
        df = DataFrame()
        sql = "SELECT ? FROM ? WHERE DATE BETWEEN DATE(sdate) AND DATE(edate)"

    def set_input_value(self, id, value):
        self.dynamicCall("SetInputValue(QString, QString)", id, value)
    def load_stock_data(self):
        start_date =

    '''

    # 추가할 것 : 사용자가 선택한 데이터 종류에 따라 데이터 보여주는 기능
    def setStockTable(self, sdate, edate, col):
        print("flag1234567")
        cursor4 = self.con.cursor()
        print("flagisdijfsjdfjsa;fja")
        sql = "SELECT * FROM SamsungElectronics WHERE DATE(Date) BETWEEN DATE(?) AND DATE(?)"
        print("flag4")
        cursor4.execute(sql, (sdate, edate))
        print("flag5")
        self.con.commit()
        print("flag6")
        #df = pd.read_sql("select * from stock", self.con, index_col=None)
        print("flag7")

        data = cursor4.fetchall()

        self.stock_tableWidget.setRowCount(1000)
        self.stock_tableWidget.setColumnCount(5)
        cnt_row = len(data)
        cnt_col = len(data[0])

        i = 0
        j = 0
        while i<cnt_row:
            while j<cnt_col:
                item = QTableWidgetItem(str(data[i][j]))
                self.stock_tableWidget.setItem(i, j, item)
                j = j+1
            i = i+1

        cursor4.close()

    # 주식 데이터 조회 버튼 이벤트
    def stock_btn_clicked(self):
        sdate = self.stock_sdate.text()
        edate = self.stock_edate.text()
        print("flag1")
        # 가져올 컬럼의 리스트
        type = []
        # 날짜는 디폴트로 무조건 가져오기
        type.append('Date')
        print("flag2")
        # 체크박스에 체크됐으면 type리스트에 붙이기
        if self.check_close.isChecked() == True:
            type.append('Close')
        if self.check_open.isChecked() == True:
            type.append('Open')
        if self.check_high.isChecked() == True:
            type.append('High')
        if self.check_low.isChecked() == True:
            type.append('Low')
        print("flag3")

        # 위젯에 갖다붙이는 함수 호출!
        self.setStockTable(sdate, edate, type)



if __name__=="__main__":
    app = QApplication(sys.argv)
    pyquery = Mywindow()
    pyquery.show()
    app.exec_()




