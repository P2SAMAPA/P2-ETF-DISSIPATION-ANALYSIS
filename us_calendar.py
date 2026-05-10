import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
from pandas.tseries.offsets import CustomBusinessDay

class NYSECalendar(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]

nyse_bd = CustomBusinessDay(calendar=NYSECalendar())

def next_trading_day(from_date=None):
    if from_date is None:
        from_date = pd.Timestamp.today()
    else:
        from_date = pd.Timestamp(from_date)
    return (from_date + nyse_bd).date()
