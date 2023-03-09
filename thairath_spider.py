import scrapy
from scrapy.selector import Selector
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time




class Thairath1Spider(scrapy.Spider):
    name = "thairath1"
    allowed_domains = ["www.thairath.co.th"]
    start_urls = ["https://www.thairath.co.th/column/newspaper"]

    def __init__(self): 
        chrome_options = Options()
        driver = webdriver.Chrome(executable_path=str('./chromedriver'), options=chrome_options)
        driver.get("https://www.thairath.co.th/column/newspaper")

        wait = WebDriverWait(driver, 10)
        
        i = 0
        while i < 900:
            try:
                time.sleep(2)
                element = wait.until(EC.visibility_of_element_located(
                    (By.XPATH, "(//div[@class='__Column_Newspaper_ListA_ViewAllButton css-dvgxrf efr6tej37'][1]/a)"))) # //*[@id="__next"]/main/div/div[3]/div[2]/div/div[4]
                element.click() 
                i += 1
            except TimeoutException:
                break
                
        self.html = driver.page_source # copy down all that's now shown on the page


    def parse(self, response):
        resp = Selector(text=self.html)
        results = resp.xpath("//div[@class='css-hxcsaq e1w36dl217']")
        n = 1
        for result in results:
            title = result.xpath(".//h2[@class='css-11fy7ft efr6tej2']/a//text()").get()
            link = result.xpath(".//h2[@class='css-11fy7ft efr6tej2']/a//@href").get()
            filout = result.xpath(".//div[@class='css-gy6boz e1w36dl219']//div[@class='css-10iury e1w36dl220']/a[2]/text()").get() # class='css-10iury e1w36dl220'

            if filout == 'ไฮไฟ': 
                continue # filter page with no content out
            else:
                yield response.follow(url=link, callback=self.parse_article, meta={"id":n, "title": title})
                n = n+1

    
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
