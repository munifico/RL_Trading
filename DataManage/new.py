import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from Project.Webreader2 import Webreader2
from Project.Kiwoom import Kiwoom
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from pandas import DataFrame

con = sqlite3.connect("c:/Users/Eugene/pyquery.db")
cursor = con.cursor()
sqldhfjsdjfafjdsjfjkjfds