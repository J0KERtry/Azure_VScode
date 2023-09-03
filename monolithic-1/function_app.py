import azure.functions as func
import logging
from collections import Counter
import math
import copy
import random
import time
del_count = 10 # 削除するマスの数を設定

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# パラメータ取得関数
def get_param_from_request(req, param_name, json_key=None):
    row = req.params.get(param_name)
    if row is None:
        try:
            row = req.get_json().get(json_key or param_name)
        except (ValueError, KeyError):
            row = 1
    return int(row)

##################
## クライアント ##
##################
@app.route(route="main")
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # パラメータかjsonから値取得
    row = get_param_from_request(req, "row")

    # 関数呼び出し
    result = execute_process(row)

    return func.HttpResponse(result)


##############
## 実行部分 ##
##############
def execute_process(row_count: int) -> str:
    # 解けない問題であれば繰り返し生成 -> 解く
    start = time.time()   # 実行開始時間保存
    data = prepare_block_check(row_count)   # 対角線上に数字が含まれているパネル(data)を作成
    repeate = 1   # 問題を生成した回数
    solver(data, 0, 0, row_count)   # 対角線上に数字が含まれているパネル(data)を作成
    while any(0 in row for row in data):
        data = prepare_block_check(row_count)
        repeate += 1
        solver(data, 0, 0, row_count)
    performance_time = time.time() - start

    result = []  # 結果の文字列を格納するリスト
    result.append("数独を生成した回数：" + str(repeate) + "\n")
    result.append("実行時間 : " + str(performance_time) + "\n")
    for a in data:
        result.append(" ".join(map(str, a)))
    result.append("")  # 空行を追加

    draw_out(data, del_count, row_count)  # 要素を削除
    result.append("生成した例題")
    for a in data:
        result.append(" ".join(map(str, a)))

    return "\n".join(result)  # 改行を挟んでリストを文字列に変換して返す

#########################
##  アクティビティ関数  ##
#########################
# 重複なしのランダムを生成し、リストで返す
def random_int(a: int, b: int, k: int) -> list:
    ns = []
    while len(ns) < k:
        n = random.randint(a, b)
        if n not in ns:
            ns.append(n)
    return ns

# 対角線上のブロックにランダムな数字格納
def prepare_block_check(row_count: int) -> list:
  panel = [[0 for i in range(row_count)] for j in range(row_count)] # 数独用のパネル
  random = []   # 乱数を格納
  mass = int(math.sqrt(row_count))
  for i in range(mass):
    xbase = mass * i   # 対角線上の開始点が panel[ybase][xbase]となる
    ybase = mass * i
    random = random_int(1, row_count, row_count)
    for y in range(ybase, ybase + mass):
      for x in range(xbase, xbase + mass):
        panel[y][x] = random[(y - ybase)*mass + (x - xbase)]
  return panel

# 横をチェック
def row_check(values: list, y: int, i: int, row_count: int) -> bool:
    return all(True if i != values[y][_x] else False for _x in range(row_count))

# 縦をチェック
def column_check(values: list, x: int, i: int, row_count: int) -> bool:    
    return all(True if i != values[_y][x] else False for _y in range(row_count))

# row x rowのブロックをチェック
def block_check(values: list, x: int, y: int, i: int, row_count: int) -> bool:
    mass = int(math.sqrt(row_count))
    xbase = (x // mass) * mass
    ybase = (y // mass) * mass
    return all(True if i != values[_y][_x] else False
            for _y in range(ybase, ybase + mass)
                for _x in range(xbase, xbase + mass))

# すべてのチェックを満たしているかチェック
def check(values: list, x: int, y: int, i: int, row_count: int) -> bool:
    return all([row_check(values, y, i, row_count), 
                column_check(values, x, i, row_count), 
                block_check(values, x, y, i, row_count)])

# 入れられる数の少ないリスト番号を返す
def count(values: list, row_count: int) -> tuple:
  min_val = row_count
  min_x, min_y = -1, -1   # 候補の個数が少ない座標を保存
  result = []   # 入る候補の数字を保存

  for y in range(row_count):
    for x in range(row_count):
      if values[y][x] == 0:  # マスが0のとき
        panel = []
        for i in range(1, row_count+1):
          if check(values, x, y, i, row_count): # 入れられる数字を追加
            panel.append(i)
        if (min_val > len(panel)): # 入れられる候補の少ないものが見つかったら更新
          min_val = len(panel)
          result = copy.copy(panel)
          min_x, min_y = x, y

  return min_x, min_y, result

# 数独を解く関数
def solver(values: list, x: int, y: int, row_count: int) -> bool:
  min_x, min_y, panel = count(values, row_count)  # 入れられる数の少ないインデックス取得

  if (min_x==-1): #終了条件
    return True
  for i in range(0, len(panel)):
    values[min_y][min_x] = panel[i]  #数字を入れる
    if solver(values, min_x, min_y, row_count):
      return True
    values[min_y][min_x] = 0 #戻ってきたら0に戻す    

  return False

# 完成している数独から要素を削除する
def draw_out(values: list, del_count: int, row_count: int) -> list:
    draw_out_index = []  
    draw_out_index = random_int(0, row_count * row_count - 1, del_count)  # 抜き出す配列の要素番号を取得
    for i in range(del_count):
        y = draw_out_index[i] // row_count
        x = draw_out_index[i] % row_count
        values[y][x] = 0
    return values

