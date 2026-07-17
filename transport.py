import pandas as pd
import numpy as np
trains = {
    "G1001": {
        "start": "上海",
        "end": "南京",
        "price": 145,
        "seat": 30
    },
    "G1002": {
        "start": "上海",
        "end": "杭州",
        "price": 95,
        "seat": 50
    },
    "G1003": {
        "start": "上海",
        "end": "北京",
        "price": 680,
        "seat": 20
    }
}
orders = []
def show_menu():
    """显示主菜单"""
    print("高铁订票系统")
    print("1. 查看所有车次")
    print("2. 查询车次")
    print("3. 购买车票")
    print("4. 查看我的订单")
    print("5. 退票")
    print("6. 统计收入")
    print("7. 退出")
    print("================================")


def show_trains():
    """查看所有车次"""
    print("\n车次\t起点\t终点\t票价\t剩余座位")
    for train_id, info in trains.items():
        print(f"{train_id}\t{info['start']}\t{info['end']}\t{info['price']}\t{info['seat']}")

def search_train():
    """按终点查询车次"""
    end_city = input("请输入终点：")
    found = False
    print("\n车次\t起点\t终点\t票价\t剩余座位")
    for train_id, info in trains.items():
        if info["end"] == end_city:
            print(f"{train_id}\t{info['start']}\t{info['end']}\t{info['price']}\t{info['seat']}")
            found = True
    if not found:
        print(f"没有找到开往{end_city}的列车")

def buy_ticket():
    """购买车票"""
    name = input("请输入姓名：")
    id_card = input("请输入身份证：")
    train_id = input("请输入车次：")
    count = int(input("请输入购买数量："))
    
    # 检查车次是否存在
    if train_id not in trains:
        print("车次不存在！")
        return
    
    train_info = trains[train_id]
    # 检查余票是否足够
    if train_info["seat"] < count:
        print(f"余票不足！剩余座位：{train_info['seat']}")
        return
    
    # 扣减座位
    train_info["seat"] -= count
    # 计算金额
    total_money = train_info["price"] * count
    # 保存订单
    order = {
        "name": name,
        "id": id_card,
        "train": train_id,
        "count": count,
        "money": total_money
    }
    orders.append(order)
    print(f"购票成功！共{count}张，总金额{total_money}元")

def show_orders():
    """查看我的订单"""
    if not orders:
        print("暂无订单")
        return
    print("\n姓名\t车次\t数量\t金额")
    for order in orders:
        print(f"{order['name']}\t{order['train']}\t{order['count']}张\t{order['money']}元")

def refund_ticket():
    """退票"""
    id_card = input("请输入身份证号：")
    found_order = None
    for order in orders:
        if order["id"] == id_card:
            found_order = order
            break
    
    if not found_order:
        print("未找到对应订单")
        return
    
    # 恢复座位库存
    trains[found_order["train"]]["seat"] += found_order["count"]
    # 删除订单
    orders.remove(found_order)
    print(f"退票成功！{found_order['train']} {found_order['count']}张票已退回")

def count_money():
    """统计收入与库存"""
    total_orders = len(orders)
    total_money = sum(order["money"] for order in orders)
    total_seat = sum(info["seat"] for info in trains.values())
    
    print(f"\n今日订单：{total_orders}")
    print(f"销售额：{total_money}")
    print(f"剩余座位：{total_seat}")

# 主程序循环
def main():
    while True:
        show_menu()
        choice = input("请输入选项：")
        if choice == "1":
            show_trains()
        elif choice == "2":
            search_train()
        elif choice == "3":
            buy_ticket()
        elif choice == "4":
            show_orders()
        elif choice == "5":
            refund_ticket()
        elif choice == "6":
            count_money()
        elif choice == "7":
            print("感谢使用，退出系统")
            break
        else:
            print("输入错误，请重新选择")

if __name__ == "__main__":
    main()