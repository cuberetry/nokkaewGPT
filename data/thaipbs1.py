import scrapy
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from scrapy.selector import Selector
import time


class Thaipbs1Spider(scrapy.Spider):
    name = "thaipbs1"
    allowed_domains = ["www.thaipbs.or.th"]
    start_urls = ["https://www.thaipbs.or.th/news/archive/2023-03-08"]


    # initiating selenium
    def __init__(self):
        
        # set up the driver
        chrome_options = Options()
        # chrome_options.add_argument("--headless") # uncomment if don't want to appreciate the sight of a possessed browser
        driver = webdriver.Chrome(executable_path=str('./chromedriver'), options=chrome_options)
        driver.get("https://www.thaipbs.or.th/news/archive/2023-03-08")

        # record the first page
        self.html = [driver.page_source]
        
        # start turning pages
        i = 0
        while i < 100: # 100 is just right to get us back to July
            i += 1
            time.sleep(5) # just in case the next button hasn't finished loading
            next_btn = driver.find_element_by_xpath("(//div[contains(@class, 'pagination-arrow')])[2]") # click next button
            next_btn.click()
            self.html.append(driver.page_source) # not the best way but will do
            

    def parse(self, response):
        n = 7021
        for page in self.html:
            resp = Selector(text=page)
            results = resp.xpath("//div[@class='cnn-search__result cnn-search__result--article']/div/h3/a") # result iterator
            for result in results:
                title = result.xpath(".//text()").get()
                if ("Video" in title) | ("coronavirus news" in title) | ("http" in title):
                    continue # ignore videos and search-independent news or ads
                else:
                    link = result.xpath(".//@href").get()[13:] # cut off the domain; had better just use request in fact
                    yield response.follow(url=link, callback=self.parse_article, meta={"id":n,"title": title})

    def parse_article(self, response):
        n = response.request.meta['id']
        title = response.request.meta['title']
        # body = response.css('div[itemprop="articleBody"] p')
        highlight_ex = ''.join(response.css('div[itemprop="articleBody"] p')[0].css('*::text').extract())
        highlight = highlight_ex.replace('SPONSORED', ' ')

        content_ex = ''.join(response.css('div[itemprop="articleBody"] p')[1:].css('*::text').extract())
        content = content_ex.replace('SPONSORED', ' ')
        
        if (content != '') and (len(content)>100) and (len(highlight)>100): 
            yield {
                "id": n,
                "title": title,
                "highlight": highlight,
                "content": content
            }
            n = n+1

