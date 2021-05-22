




class Exchange(object):

	def __init__(self, name):
		self.name = name
		

	def stats(self):
		# Print portefolio stats
		pass





class Kracken(Exchange):

	def __init__(self):
		super("kracken")
		pass


class CoinbasePro(Exchange):

	def __init__(self):
		super("coinbasepro")
		pass

class Binance(Exchange):

	def __init__(self, exchange):
		super("binance")
		pass
