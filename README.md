Playing around with Authorship Attribution and Morrissey 
========================
Dependencies:
- scrapy 
- scikit-learn 
- nltk

Files:
- lyrics-scraper.py - slightly hardcoded scrapy spider class for scraping all morrissey and the smiths lyrics
- similar-scraper-combo.py - hardcoded scrapy spider class for scraping all the lyrics from the smiths related artists
- learn.py - where the basic, bag of words, lexical feature, supervised learning magic happens; the feature union pipeline has the option of using rank distance too; some unigram feature selection with ANOVA also
- ranker.py - file with all the rank distance related code
