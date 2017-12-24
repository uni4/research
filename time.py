#プログラムの動作時間を計測するプログラム
#https://www.sejuku.net/blog/23955

import time

t1 = time.time() 
 
# 計測したい処理
for i in range(1000000):
    i ** 10
 
# 処理後の時刻
t2 = time.time()
 
# 経過時間を表示
elapsed_time = t2-t1
print(f"経過時間：{elapsed_time}")