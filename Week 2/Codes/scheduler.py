import schedule
import time
from train_model import train_model

schedule.every(10).seconds.do(train_model)


while True:
    schedule.run_pending()
    time.sleep(1)