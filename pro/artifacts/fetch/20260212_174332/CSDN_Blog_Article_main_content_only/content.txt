前些天发现了一个人工智能学习网站，内容深入浅出、易于理解。如果对人工智能感兴趣，不妨点击查看。

之前的代码一直报错521，不清楚什么原因

因此重新分析整个过程，并对代码进行更新

结果如图

参考：

批量获取CSDN文章对文章质量分进行检测，有助于优化文章质量

【python】我用python写了一个可以批量查询文章质量分的小项目（纯python、flask+html、打包成exe文件）

`import json import pandas as pd from openpyxl import Workbook, load_workbook from openpyxl.utils.dataframe import dataframe_to_rows import math import requests # 批量获取文章信息并保存到excel class CSDNArticleExporter: def __init__(self, username, cookies, Referer, page, size, filename): self.username = username self.cookies = cookies self.Referer = Referer self.size = size self.filename = filename self.page = page def get_articles(self): url = "https://blog.csdn.net/community/home-api/v1/get-business-list" params = { "page": {self.page}, "size": {self.size}, "businessType": "blog", "username": {self.username} } headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3', 'Cookie': self.cookies, # Setting the cookies string directly in headers 'Referer': self.Referer } try: response = requests.get(url, params=params, headers=headers) response.raise_for_status() # Raises an HTTPError if the response status code is 4XX or 5XX data = response.json() return data.get('data', {}).get('list', []) except requests.exceptions.HTTPError as e: print(f"HTTP错误: {e.response.status_code} {e.response.reason}") except requests.exceptions.RequestException as e: print(f"请求异常: {e}") except json.JSONDecodeError: print("解析JSON失败") return [] def export_to_excel(self): df = pd.DataFrame(self.get_articles()) df = df[['title', 'url', 'postTime', 'viewCount', 'collectCount', 'diggCount', 'commentCount']] df.columns = ['文章标题', 'URL', '发布时间', '阅读量', '收藏量', '点赞量', '评论量'] # df.to_excel(self.filename) # 下面的代码会让excel每列都是合适的列宽，如达到最佳阅读效果 # 你只用上面的保存也是可以的 # Create a new workbook and select the active sheet wb = Workbook() sheet = wb.active # Write DataFrame to sheet for r in dataframe_to_rows(df, index=False, header=True): sheet.append(r) # Iterate over the columns and set column width to the max length in each column for column in sheet.columns: max_length = 0 column = [cell for cell in column] for cell in column: try: if len(str(cell.value)) > max_length: max_length = len(cell.value) except: pass adjusted_width = (max_length + 5) sheet.column_dimensions[column[0].column_letter].width = adjusted_width # Save the workbook wb.save(self.filename) class ArticleScores: def __init__(self, filepath): self.filepath = filepath @staticmethod def get_article_score(article_url): url = "https://bizapi.csdn.net/trends/api/v1/get-article-score" # TODO: Replace with your actual headers headers = { "Accept": "application/json, text/plain, */*", "X-Ca-Key": "203930474", "X-Ca-Nonce": "b35e1821-05c2-458d-adae-3b720bb15fdf", "X-Ca-Signature": "gjeSiKTRCh8aDv0UwThIVRITc/JtGJkgkZoLVeA6sWo=", "X-Ca-Signature-Headers": "x-ca-key,x-ca-nonce", "X-Ca-Signed-Content-Type": "multipart/form-data", } data = {"url": article_url} try: response = requests.post(url, headers=headers, data=data) response.raise_for_status() # This will raise an error for bad responses return response.json().get('data', {}).get('score', 'Score not found') except requests.RequestException as e: print(f"Request failed: {e}") return "Error fetching score" def get_scores_from_excel(self): df = pd.read_excel(self.filepath) urls = df['URL'].tolist() scores = [self.get_article_score(url) for url in urls] return scores def write_scores_to_excel(self): df = pd.read_excel(self.filepath) df['质量分'] = self.get_scores_from_excel() df.to_excel(self.filepath, index=False) if __name__ == '__main__': total = 10 #已发文章总数量 # TODO:调整为你自己的cookies，Referer，CSDNid, headers cookies = 'uuid_tt_dd=10' # Simplified for brevity Referer = 'https://blog.csdn.net/WTYuong?type=blog' CSDNid = 'WTYuong' t_index = math.ceil(total/100)+1 #向上取整，半闭半开区间，开区间+1。 # 获取文章信息 # CSDNArticleExporter("待查询用户名", 2（分页数量，按总文章数量/100所得的分页数）,总文章数量仅为设置为全部可见的文章总数。 # 100（最大单次查询文章数量不大于100）, 'score1.xlsx'（待保存数据的文件，需要和下面的一致）) for index in range(1,t_index): #文章总数 filename = "score"+str(index)+".xlsx" exporter = CSDNArticleExporter(CSDNid, cookies, Referer, index, 100, filename) # Replace with your username exporter.export_to_excel() # 批量获取质量分 score = ArticleScores(filename) score.write_scores_to_excel()`

python运行

浏览器访问需要获取文章的博主首页地址，并且打开开发者工具快捷键F12

然后点击网络选项，我们在刷新页面可以看到发送的请求地址。

经过测试，请求头只需要包括Cookies、Referer参数即可。

关于如何获取cookie：

先去质量查询地址：https://www.csdn.net/qc

然后输入任意一篇文章地址进行查询，同时检查页面，在Network选项下即可看到调用的API的请求地址、请求方法、请求头、请求体等内容：

经过测试，请求头只需要下面这几个参数即可。

请求头分析

X-Ca-Key:使用自己浏览器的

X-Ca-Nonce:使用自己浏览器的

X-Ca-Signature:使用自己浏览器的

X-Ca-Signature-Headers:x-ca-key,x-ca-nonce

X-Ca-Signed-Content-Type:multipart/form-data

Accept :application/json, text/plain, */*