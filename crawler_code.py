import requests
from bs4 import BeautifulSoup
import time
import random
import csv
import os
from fake_useragent import UserAgent

output_dir = 'D:\OneDrive\桌面\当当网数据爬取'  # 指定文件夹路径
os.makedirs(output_dir, exist_ok=True)  # 创建文件夹(如果不存在)

# 修改CSV文件路径部分
csv_path = os.path.join(output_dir, 'dangdang_books.csv')

# 初始化UserAgent和CSV文件
# 初始化
ua = UserAgent()
headers = {'User-Agent': ua.random}
base_url = 'http://search.dangdang.com/?key=%CA%E9%BC%AE&act=input&page_index='

# CSV文件头
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['书名', '价格', '原价', '折扣', '作者', '出版社', '出版日期',
                    '评论数', '评分', 'ISBN', '详情页链接'])

# 确保文件写入方式正确
def init_csv_file():
    csv_path = os.path.join(output_dir, 'dangdang_books.csv')
    # 强制重新创建文件并写入表头（无论文件是否存在）
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['书名', '价格', '原价', '折扣', '作者', '出版社',
                        '出版日期', '评论数', '评分', 'ISBN', '详情页链接'])


# 在爬取前初始化CSV文件
init_csv_file()

def crawl_dangdang(page):
    url = f'http://search.dangdang.com/?key=%CA%E9%BC%AE&act=input&page_index={page}'
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # 调试：打印页面内容检查是否获取成功
        print(f"正在解析第{page}页，状态码:{response.status_code}")

        # 修改选择器 - 使用更通用的匹配方式
        books = soup.select(
            'ul.bigimg li'
        )  # 或者尝试 soup.find_all('li', class_=lambda x: x and 'line' in x)

        if not books:
            # 保存错误页面供调试
            with open(f'error_page_{page}.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"⚠️ 第{page}页未找到数据，已保存页面供分析")
            return

        for book in books:
            # 基础信息
            title = book.find('a', attrs={
                'dd_name': '单品标题'
            })['title'] if book.find('a', attrs={'dd_name': '单品标题'}) else '无'
            price = book.find('p', class_='price').find(
                'span',
                class_='search_now_price').text.strip('¥') if book.find(
                    'p', class_='price') else '无'
            original_price = book.find(
                'span',
                class_='search_pre_price').text.strip('¥') if book.find(
                    'span', class_='search_pre_price') else price
            discount = round(float(price) / float(original_price) *
                             10, 1) if original_price != price else 10.0

            # 作者出版社信息
            # 改进的作者/出版社信息提取
            author_info = book.find('p',
                                    class_='search_book_author') or book.find(
                                        'span', class_='search_book_author')
            if author_info:
                author_parts = [
                    part.strip() for part in author_info.text.split('/')
                    if part.strip()
                ]
                author = author_parts[0] if len(author_parts) > 0 else '无'
                publisher = author_parts[-2] if len(author_parts) > 2 else '无'
                pub_date = author_parts[-1] if len(author_parts) > 1 else '无'
            else:
                author = publisher = pub_date = '无'


            # 评分和评论
            comments = book.find(
                'a', class_='search_comment_num').text.strip() if book.find(
                    'a', class_='search_comment_num') else '0'
            rating_span = book.find('span', class_='search_star_black')
            rating = rating_span.find('span')['style'].split(
                ':')[-1].strip() if rating_span else '0'

            # 新增特征
            detail_link = book.find('a', title=True)['href']
            isbn = detail_link.split('/')[-1].split(
                '.')[0] if detail_link else '无'
            category = book.find(
                'span',
                class_='search_book_catalog').text.strip() if book.find(
                    'span', class_='search_book_catalog') else '无'

            # 写入CSV
            with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    title, price, original_price, discount, author,pub_date,
                    publisher, comments, rating, isbn, detail_link
                ])

        print(f'第{page}页爬取完成，共{len(books)}条数据')
        time.sleep(random.uniform(2, 4))  # 增加随机延迟范围

    except Exception as e:
        print(f'第{page}页爬取失败:', e)

# 爬取60页数据确保3600样本(每页约60本)
for page in range(1, 61):
    crawl_dangdang(page)
    if page % 10 == 0:  # 每10页更换User-Agent
        headers['User-Agent'] = ua.random
