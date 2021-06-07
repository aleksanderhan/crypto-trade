from queue import PriorityQueue



sales = PriorityQueue()


bought = coins_bought
while bought > 0:
    if not sales.empty():
        sold_price, sold_amount = sales.get() # NB! negative values
        if bought > abs(sold_amount):
            reward += sold_price * sold_amount - bought * current_price
            bought += sold_amount
        elif bought < abs(sold_amount):
            reward += sold_price * sold_amount - bought * current_price
            sales.put((sold_price, sold_amount + bought))
            bought = 0
        else:
            bought = 0
    else:
        reward = -bought * current_price
        bought = 0