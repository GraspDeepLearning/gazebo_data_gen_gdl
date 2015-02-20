import time

def get_date_string():
    t = time.localtime()
    minute = str(t.tm_min)
    if len(minute) == 1:
        minute = '0' + minute
    t_string = str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + str(t.tm_hour) + "_" + minute
    return t_string