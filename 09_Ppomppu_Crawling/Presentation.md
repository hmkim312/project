## 뽐뿌 특가 게시판 크롤링
---

### 1. 개요
특가 정보에 관심이 많은 사람으로써 **특가 데이터 분석**을 위해 뽐뿌의 특가 게시판을 크롤링 하여 특가 데이터를 확보하였다. **특가 데이터 분석**은 `1)데이터 확보` `2)특가 데이터 분석` `3)특가 데이터 예측 모델링` 순으로 진행 된다. 
대부분의 사람들은 현명한 소비를 하기 위해 노력한다. 인터넷이 발달하기 전에는 '발품'을 팔아가며 같은 가격이라면 더 좋은 품질의 상품을, 혹은 같은 상품이라면 더 저렴하게 구매하기 위해 노력했다. 인터넷과 물류 유통의 발달로 이제는 '발품'을 넘어 인터넷상에서 더  저렴한 제품을 찾는 '손품'을 파는 사람이 늘어나기 시작했다. 이러한 틈새를 노려 각종 커뮤니티에는 사용자 유입을 위해 '특가를 공유 하는 게시판'을 운영중이다. '뽐뿌'도 이러한 커뮤니티 중 하나이며 특가 관련 정보가 활발히 공유되는 오래된 커뮤니티이다. 특가 게시물을 분석하기에 앞서 데이터를 구하기 위해 뽐뿌에 올라오는 특가 게시물을 크롤링 하기로 결정하였다. 이 문서는 '뽐뿌 커뮤니티'의 특가 정보를 공유 하는 '뽐뿌 게시판'을 크롤링 하는 방법에 대해 설명한다.

### 2. 크롤링을 하기에 앞서
크롤링이란 웹상에 있는 데이터를 긁어오는 것을 말한다. 다만, 웹페이지의 주인에게 크롤링을 허락을 맡은게 아니기 때문에 크롤링을 진행하기 전에 **데이터를 긁어와도 되는지** 확인해 보아야한다. 대부분의 웹사이트는 URL의 맨 끝에 `/robots.txt`를 붙인 특정 페이지를 통해 크롤링 허용 여부를 알려주고 있다.
뽐뿌도 마찬가지로 [뽐뿌 크롤링 허용 정보](https://www.ppomppu.co.kr/robots.txt)에 접속하면 크롤링 허용 여부를 알수 있다.
뽐뿌는 **모든 유저 (User-agent: \*)** 대해 **zboard는 허용 (Allow\zboard)** 된 상태임을 확인하였으며 `특가 게시판`은 **비허용 (Disallow)** 목록에 없으므로 허용한것으로 판단하여, 크롤링을 하기로 결정하였다.

### 3. 크롤링
<img width="916" alt="스크린샷 2023-07-29 오전 9 31 52" src="https://github.com/hmkim312/project/assets/60168331/a00757ce-f0f3-4231-a587-e85d185601cd">

크롤링을 하기로 결정하였다면, 크롤링을 통해 어떤 데이터를 긁어올지 결정하여야 한다.
- 1) 게시글 번호
- 2) 작성자
- 3) 인기/핫 게시물 여부
- 4) 게시글 제목
- 5) 게시글에 달린 댓글 수
- 6) 게시글 등록일
- 7) 추천수
- 8) 반대수
- 9) 조회수
- 10) 게시글의 카테고리
- 11) 게시글 URL

```python
# 패키지 import
import requests
import pandas as pd
import bs4
import time
import tqdm

from datetime import datetime
from bs4 import BeautifulSoup

# 뽐뿌 게시판 크롤링
def get_datas(items):
    """ 뽐뿌 게시판 크롤링 함수
    뽐뿌 사이트 내 뽐뿌 게시판에 작성된 특가 게시글 크롤링
    Args:
        items - BS4를 사용한 HTML 파서 데이터. 뽐뿌 게시판 1페이지의 게시글 정보가 담긴 데이터
    Returns:
        DataFrame : 아래의 컬럼 정보를 가진 DataFrame을 Return
            - item_no : 게시글 번호
            - writet : 작성자
            - title : 게시글 제목
            - end : 게시글 종료 여부
            - comment : 게시글에 달린 댓글 수
            - date : 게시글 등록일
            - recommend : 추천수
            - opposite : 반대수
            - view : 조회수
            - category : 게시글의 카테고리 e.g) 디지털
            - ULR : 게시글 접속 URL
    """
    data = {
        "item_no": [],
        "writer": [],
        "title": [],
        "end": [],
        "comment": [],
        "date": [],
        "recommend": [],
        "opposite": [],
        "view": [],
        "category": [],
        "URL": [],
    }

    for item in items:
        item_no = item.find("td", "eng list_vspace").text.strip()
        data["item_no"].append(item_no)

        writer = item.find("span", "list_name")
        data["writer"].append(writer.text if writer else "Image_name")

        title = item.find("font")
        data["title"].append(title.text if title else "해당글은 게시중단요청에 의해 블라인드 처리된 글입니다.")

        data["end"].append(bool(item.find("img", {"src": "/zboard/skin/DQ_Revolution_BBS_New1/end_icon.PNG"})))

        comment = item.find("span", "list_comment2")
        data["comment"].append(comment.text.strip() if comment else 0)

        data["date"].append(item.find_all("td", "eng list_vspace")[1]["title"])

        rec_opp = item.find_all("td", "eng list_vspace")[2].text.split('-')
        data["recommend"].append(rec_opp[0] if len(rec_opp) > 1 else 0)
        data["opposite"].append(rec_opp[1] if len(rec_opp) > 1 else 0)

        data["view"].append(item.find_all("td", "eng list_vspace")[3].text)

        cat = item.find("span", {"style": "color:#999;font-size:11px;"})
        data["category"].append(cat.text if cat else "")

        data["URL"].append(f"https://www.ppomppu.co.kr/zboard/view.php?id=ppomppu&no={item_no}")

    return pd.DataFrame(data)

# 파라미터
end_page = 5900
base_url = "https://www.ppomppu.co.kr/zboard/zboard.php?id=ppomppu&page={}"
temp_list = []

# 시작 시간 체크
start_time = time.time()

# 크롤링 실행
for page in tqdm.tqdm(range(2, end_page)):
    response = requests.get(base_url.format(page))
    
    if response.status_code == 200:
        html = BeautifulSoup(response.text, 'html.parser')
        items = html.find_all("tr", ["common-list0", "common-list1"])
        temp_df = get_datas(items)
        temp_list.append(temp_df)
        time.sleep(0.1)

# 최종 데이터 프레임 저장
df = pd.concat(temp_list).reset_index(drop=True)

# 소요시간 체크
elapsed_time = time.time() - start_time
print(f"Crawling took {round(elapsed_time, 2)} seconds")

# 인기/핫 게시물 크롤링
# 파라미터
end_page = 1600
hot_url = "https://www.ppomppu.co.kr/zboard/zboard.php?id=ppomppu&page={}&hotlist_flag=999"
temp_list = []

# 시작 시간 체크
start_time = time.time()

# Crawling
for page in tqdm.tqdm(range(2, end_page)):
    response = requests.get(hot_url.format(page))
    
    if response.status_code == 200:
        html = BeautifulSoup(response.text, 'html.parser')
        items = html.find_all("tr", ["common-list0", "common-list1"])
        temp_df = get_pop_post(items)
        temp_list.append(temp_df)
        # time.sleep(0.1)

# 최종 데이터 프레임 저장
hot_df = pd.concat(temp_list).reset_index(drop=True)

# 소요시간 체크
elapsed_time = time.time() - start_time
print(f"Crawling took {round(elapsed_time, 2)} seconds")

df = df.merge(hot_df, how="left").fillna(False)

# 데이터 csv 저장
now = str(datetime.now())
df.to_csv(f"./datas/{now}_{len(df)}개.csv", index=False)
```

