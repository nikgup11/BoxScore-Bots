from nbainjuries import injury
from datetime import datetime

today = datetime.now()
today_for_injury = datetime(year=today.year, month=today.month, day=today.day, hour=11, minute=0)

df_output = injury.get_reportdata(today_for_injury, return_df=True)
df_output.to_csv("./data-collection/clean_data/injuries.csv", index=False)