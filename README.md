Playing around with Authorship Attribution and Morrissey 
========================
Dependencies:
- scrapy 
- scikit-learn 
- nltk

Files:
- lyrics-scraper.py - slightly hardcoded scrapy spider class for scraping all morrissey and the smiths lyrics
- similar-scraper-combo.py - hardcoded scrapy spider class for scraping all the lyrics from the smiths related artists
- learn.py - where the basic, bag of words, supervised learning magic happens; also some unigram feature selection with ANOVA 
