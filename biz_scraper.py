from requests_html import HTMLSession, AsyncHTMLSession
from lib.chan_tools import *
from lib.util import log_error
import queue, threading, math

BASE_URL = 'https://archived.moe/'







class ArticleProcessor(threading.Thread):

	def __init__(self, base_url):
		super(ArticleProcessor, self).__init__()
		self.base_url = base_url
		self.simultaneous_requests = 3
		self.q = queue.Queue(maxsize=1000)
		self.asession = AsyncHTMLSession()
		self.enqueued = set([])
		self.stopped = False

	def put(self, thread_id):
		if thread_id not in self.enqueued:
			self.q.put(thread_id, block=True, timeout=None)
			self.enqueued.add(thread_id)

	def stop(self):
		self.enqueued = set([])
		self.asession.close()
		self.stopped = True
		for i in range(self.simultaneous_requests):
			self.processor.q.put(None)

	def run(self):
		while not self.stopped:
			thread_ids = [self.q.get(block=True, timeout=None) for _ in range(self.simultaneous_requests)]
			requests = [lambda url=self.base_url+str(thread_id): self._get_thread(url) for thread_id in thread_ids]
			try:
				rs = self.asession.run(*requests)
			except Exception as e:
				log_error(e)



			for r in rs:
				main = r.html.find('#main', first=True)
				articles = main.find('article')[:-1]

				op = articles[0]
				title = op.find('.post_title', first=True)
				print(title.text)

				for reply in articles[1:]:
					pass

	async def _get_thread(self, thread_url):
		return await self.asession.get(thread_url)
	



class ArchivedMoeScraper(object):

	def __init__(self, board, start_page=1, stop_page=math.inf):
		self.processor = ArticleProcessor(base_url=BASE_URL+board+'/thread/')
		self.session = HTMLSession()
		self.board = board
		self.page = start_page
		self.stop_page = stop_page

	def start(self):
		self.processor.start()

		while self.page <= self.stop_page:
			url = BASE_URL + self.board + '/page/' + str(self.page)
			try:
				r = self.session.get(url)
				main = r.html.find('#main', first=True)
				articles = main.find('article')[:-1]
				
				if (len(articles) < 1):
					self.stop()
					break

				for article in articles:
					thread_id = article.attrs['id']
					self.processor.put(thread_id)
			except Exception as e:
				log_error(e)

			self.page += 1

		print(f'Finished scraping {self.board} for thread ids.')
		self.session.close()







if __name__ == '__main__':
	biz_scraper = ArchivedMoeScraper('biz')

	biz_scraper.start()