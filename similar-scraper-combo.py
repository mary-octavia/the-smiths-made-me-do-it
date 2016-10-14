import scrapy

class SimilarArtistsSpider(scrapy.Spider):
	name = 'allmusic'
	# start_urls = ['http://www.azlyrics.com/m/morrissey.html']
	# start_urls = ['http://www.metrolyrics.com/morrissey-lyrics.html']
	# start_urls = ['http://www.metrolyrics.com/the-smiths-lyrics.html']
	start_urls = ['http://www.allmusic.com/artist/the-smiths-mn0000899530/related']
	# download_delay = 0.7
	

	def parse(self, response):
		for href in response.css('section[class*="related influencers"] ul li a::attr(href)'):
			full_url = response.urljoin(href.extract()) + "/songs/all"
			# print "!!!!!full_url!!!!! ", full_url
			yield scrapy.Request(full_url, callback=self.parse_songs)
			full_url = response.urljoin(href.extract()) + "/songs/all/2"
			yield scrapy.Request(full_url, callback=self.parse_songs)
			full_url = response.urljoin(href.extract()) + "/songs/all/3"
			yield scrapy.Request(full_url, callback=self.parse_songs)
			full_url = response.urljoin(href.extract()) + "/songs/all/4"
			yield scrapy.Request(full_url, callback=self.parse_songs)


	def parse_songs(self, response):
		for href in  response.css('.all-songs table tbody tr td .title a::attr(href)'):
			full_url = response.urljoin(href.extract())
			scrapy.Request(full_url, callback=self.parse_album)
			full_url = full_url + "/lyrics"
			# print full_url
			yield scrapy.Request(full_url, callback=self.parse_lyrics)

	def parse_album(self, response):
		self.album = response.css('td[class*="artist-album"] div[class*="title"] a::text').extract()[0]
		print self.album
		

	def parse_lyrics(self, response):
		lyrics = response.css('p[id*="hidden_without_js"]::text').extract()
		lyrics = ''.join(lyrics)
		lyrics = lyrics.strip()

		title = response.css('h1[class*="song-title"]::text').extract()[0]
		title = title.strip()
		' '.join(title.split())

		yield {
			'band': response.css('h2[class*="song-artist"] span a::text').extract()[0],
			'title': title,
			# 'album': self.album,
			'lyrics': lyrics,
			# 'writers': response.css('h3[class*="song-composers"] a::text').extract()[0],
			'link': response.url,
		}