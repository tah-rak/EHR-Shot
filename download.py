import requests
import os

# 原始链接
url = "https://usfedu-my.sharepoint.com/personal/seungbae_usf_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fseungbae%5Fusf%5Fedu%2FDocuments%2FTeaching%2FCIS4930%2DCAI5155%5FFall25%2FProject%2FEHRShot%5Fsampled%5F2000patients%2Ezip"

# 尝试获取直接下载链接
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

response = session.get(url, allow_redirects=True)

# 检查是否是 zip 文件
print(f"Content-Type: {response.headers.get('Content-Type')}")
print(f"Content-Length: {response.headers.get('Content-Length')} bytes")

if 'zip' in response.headers.get('Content-Type', '') or response.content[:2] == b'PK':
    with open("EHRShot_sampled_2000patients.zip", "wb") as f:
        f.write(response.content)
    print("下载成功！")
else:
    print("下载的不是 zip 文件，可能需要登录")
    # 保存为 HTML 看看是什么
    with open("response.html", "w") as f:
        f.write(response.text[:1000])