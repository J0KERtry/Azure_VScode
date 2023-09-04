import azure.functions as func
import logging
from collections import Counter
from math import sqrt
from random import shuffle, randint
import time

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# パラメータ取得関数
def get_param_from_request(req, param_name, json_key=None) -> int:
    param = req.params.get(param_name)
    if param is None:
        try:
            param = req.get_json().get(json_key or param_name)
        except (ValueError, KeyError):
            param = 0
    return int(param)

##################
## クライアント ##
##################
@app.route(route="main")
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # パラメータかjsonから値取得
    row = get_param_from_request(req, "row")
    del_num = get_param_from_request(req, "del")

    # 関数呼び出し
    result = execute_process(row, del_num)

    return func.HttpResponse(result)


###########################
## オーケストレーター関数 ##
###########################
def execute_process(size: int, del_num: int) -> str:
    start = time.time()   # 実行開始時間保存
    repeate = 0   # 問題を生成した回数
    while True:  # 解けない問題であれば繰り返し生成 -> 解く
        data = prepare_block_check(size)
        repeate += 1
        if solver(data, size):
            break
    performance_time = time.time() - start

    result = []  # 画面出力を格納するリスト
    result = [f"数独を生成した回数：{repeate}\n", f"実行時間 : {performance_time}\n"]
    for a in data:
        result.append(" ".join(map(str, a)))
    result.append("")  # 空行を追加

    draw_out(data, del_num, size)  # 要素を削除
    result.append("生成した例題")
    for a in data:
        result.append(" ".join(map(str, a)))

    return "\n".join(result)  # 改行を挟んでリストを文字列に変換して返す


#########################
##  アクティビティ関数  ##
#########################
# 対角線上のブロックにランダムな数字格納
def prepare_block_check(size: int) -> list:
  panel = [[0 for i in range(size)] for j in range(size)] # 数独用のパネル
  
  mass = int(sqrt(size))
  for i in range(mass):
    xbase, ybase = mass * i, mass * i   # 対角線上の開始点が panel[ybase][xbase]となる
    numbers = list(range(1, size + 1))
    shuffle(numbers)

    for y in range(ybase, ybase + mass):
      for x in range(xbase, xbase + mass):
        panel[y][x] = numbers.pop(0)
  return panel

# 横をチェック
def row_check(values: list, y: int, i: int, size: int) -> bool:
    return all(i != values[y][_x] for _x in range(size))

# 縦をチェック
def column_check(values: list, x: int, i: int, size: int) -> bool:    
    return all(i != values[_y][x] for _y in range(size))

# row x rowのブロックをチェック
def block_check(values: list, x: int, y: int, i: int, size: int) -> bool:
    mass = int(sqrt(size))
    xbase, ybase = (x // mass) * mass,  (y // mass) * mass
    return all(i != values[_y][_x]
            for _y in range(ybase, ybase + mass)
                for _x in range(xbase, xbase + mass))

# すべてのチェックを満たしているかチェック
def check(values: list, x: int, y: int, i: int, size: int) -> bool:
    return all([row_check(values, y, i, size), 
                column_check(values, x, i, size), 
                block_check(values, x, y, i, size)])

# 入れられる数の少ないリスト番号を返す
def count(values: list, size: int) -> tuple:
  min_candidates = size
  min_x, min_y, result = -1, -1, []   # 候補の個数が少ない座標を保存, 入る候補の数字を保存

  for y in range(size):
    for x in range(size):
      if values[y][x] == 0:  # マスが0のとき
        panel = [i for i in range(1, size+1) if check(values, x, y, i, size)]  # 入れられる数字を追加
        if len(panel) < min_candidates: # 入れられる候補の少ないものが見つかったら更新
            min_candidates, min_x, min_y, result = len(panel), x, y, panel
  return min_x, min_y, result

# 数独を解く関数
def solver(values: list, size: int) -> bool:
    min_x, min_y, panel = count(values, size)  # 入れられる数の少ないインデックス取得

    if min_x == -1: #終了条件
        return True
    for i in range(len(panel)):
        values[min_y][min_x] = panel[i]  #数字を入れる
        if solver(values, size):
            return True
        values[min_y][min_x] = 0 #戻ってきたら0に戻す    

    return False

# 完成している数独から要素を削除する
def draw_out(values: list, del_count: int, size: int) -> list:
    indices = list(range(size * size))  # すべての要素のインデックスのリストを作成
    shuffle(indices)  # インデックスをシャッフル
    for i in range(del_count):
        index = indices[i]
        y = index // size
        x = index % size
        values[y][x] = 0
    return values