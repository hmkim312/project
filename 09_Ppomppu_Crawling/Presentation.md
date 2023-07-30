## 뽐뿌 특가 게시판 크롤링
---

### 1. 개요
특가 정보에 관심이 많은 사람으로써 **특가 데이터 분석**을 위해 뽐뿌의 특가 게시판을 크롤링 하여 특가 데이터를 확보하였다. **특가 데이터 분석**은 `1)데이터 확보` `2)특가 데이터 분석` `3)특가 게시물의 카테고리 예측 모델링` 순으로 진행 된다. 
대부분의 사람들은 현명한 소비를 하기 위해 노력한다. 인터넷이 발달하기 전에는 '발품'을 팔아가며 같은 가격이라면 더 좋은 품질의 상품을, 혹은 같은 상품이라면 더 저렴하게 구매하기 위해 노력했다. 인터넷과 물류 유통의 발달로 이제는 '발품'을 넘어 인터넷상에서 더  저렴한 제품을 찾는 '손품'을 파는 사람이 늘어나기 시작했다. 이러한 틈새를 노려 각종 커뮤니티에는 사용자 유입을 위해 '특가를 공유 하는 게시판'을 운영중이다. '뽐뿌'도 이러한 커뮤니티 중 하나이며 특가 관련 정보가 활발히 공유되는 오래된 커뮤니티이다. 특가 게시물을 분석하기에 앞서 데이터를 구하기 위해 뽐뿌에 올라오는 특가 게시물을 크롤링 하기로 결정하였다. 이 문서는 '뽐뿌 커뮤니티'의 특가 정보를 공유 하는 '뽐뿌 게시판'을 크롤링 하는 방법에 대해 설명한다.

### 2. 크롤링을 하기에 앞서
크롤링이란 웹상에 있는 데이터를 긁어오는 것을 말한다. 다만, 웹페이지의 주인에게 크롤링을 허락을 맡은게 아니기 때문에 크롤링을 진행하기 전에 **데이터를 긁어와도 되는지** 확인해 보아야한다. 대부분의 웹사이트는 URL의 맨 끝에 `/robots.txt`를 붙인 특정 페이지를 통해 크롤링 허용 여부를 알려주고 있다.
뽐뿌도 마찬가지로 [뽐뿌 크롤링 허용 정보](https://www.ppomppu.co.kr/robots.txt)에 접속하면 크롤링 허용 여부를 알수 있다.
뽐뿌는 **모든 유저 (User-agent: \*)** 대해 **zboard는 허용 (Allow\zboard)** 된 상태임을 확인하였으며 `특가 게시판`은 **비허용 (Disallow)** 목록에 없으므로 허용한것으로 판단하여, 크롤링을 하기로 결정하였다.

### 3. 웹 사이트 구조 파악 및 데이터 선정 


<p align="center">
    <img width="919" alt="스크린샷 2023-07-29 오후 9 28 40" src="https://github.com/hmkim312/project/assets/60168331/03abf05d-891e-41fc-81c0-25ed755fc3f4">
    <em>특가 게시물 게시판</em>
</p>


크롤링을 하기로 결정하였다면, 우선 웹 페이지가 어떤 형태로 구성되었는지 확인해보아야 한다. 뽐뿌는 크롤링하기 굉장히 쉬운 정적 웹 페이지로 이루어져있었다. 정적 웹 페이지란 서버에 미리 저장된 파일이 그대로 전달되는 웹 페이지를 말하는데, URL만 있다면 해당 페이지의 모든 데이터를 간단하게 크롤링 할 수 있다.

<p align="center">
    <img width="916" alt="스크린샷 2023-07-29 오전 9 31 52" src="https://github.com/hmkim312/project/assets/60168331/a00757ce-f0f3-4231-a587-e85d185601cd">
    <em>크롤링 데이터 선정</em>
</p>


두번째로 크롤링을 통해 어떤 데이터를 긁어올지 결정하여야 한다. 이번 **특가 데이터** 분석에 필요한 데이터는 어떤것일지 정의하여 **1) 게시글 번호**, **2) 작성자**, **3) 인기/핫 게시물 여부**, **4) 게시글 제목**, **5) 댓글 수**, **6) 등록일**, **7) 추천수**, **8) 반대수**, **9) 조회수**, **10) 게시글의 카테고리**, **11) 게시글 URL**, **12) 특가 종료 여부**를 크롤링하기로 결정하였다.

### 4. 크롤링 워크플로우


<p align="center">
    <img src = "https://github.com/hmkim312/project/assets/60168331/40bb5f4a-c313-4c84-9c1b-770d0f989f14" >
    <em>크롤링 워크플로우</em>
</p>



1. URL을 통해 서버에 데이터를 요청 한다.
  - URL에 요청을 하면 서버에서는 해당 URL에 맞는 웹페이지를 Response 해준다
2. response 받은 웹페이지에서 HTML을 파싱한다.
  - HTML 파싱이란 HTML 문서의 구조와 내용을 이해하는 과정 이다.
  - 나에게 필요한 데이터가 있는 HTML 부분을 가져온다.
3. 데이터를 크롤링 한다.
  - 크롤링할 데이터 정보를 가져오는 단계이다
  - 모든 웹 페이지를 크롤링할때까지 1번 ~ 3번의 워크플로우를 반복실행 한다
4. 데이터 저장
  - 모든 웹 페이지의 크롤링을 종료하였다면 크롤링한 데이터를 파일로 저장하거 DB에 저장하여 데이터 분석에 사용한다.

5. 데이터 아웃풋

| item_no | writer       | title                                                             | end   | coment | date             | recommend | opposite view | category       | URL                                                          | pop   | hot   |
|---------|--------------|-------------------------------------------------------------------|-------|--------|------------------|-----------|---------------|----------------|--------------------------------------------------------------|-------|-------|
| 470673  | Ko****      | [cj온스타일] 아이더 반팔 기능티 2장 (21,600원/무료)                      | True  | 8      | 23.06.29 20:39:22 | 1         | 1             | 7125 [의류/잡화] | https://www.ppomppu.co.kr/zboard/view.php?id=p... | False | False |
| 470672  | 아****       | [G마켓] PS5 디스크 에디션 갓오워 라그나로크 에디션(1218A) (606,97... | True  | 15     | 23.06.29 20:03:40 | 0         | 0             | 9811 [가전/가구] | https://www.ppomppu.co.kr/zboard/view.php?id=p... | False | False |
| 470671  | 인****     | [네이버] 국내산 1등급 소고기 등심 200G (9,900원/4000원)                | False | 8      | 23.06.29 20:00:41 | 0         | 0             | 8446 [식품/건강] | https://www.ppomppu.co.kr/zboard/view.php?id=p... | False | False |
| 470670  | Sar**** | [NS몰] 데이즈온 오한진 초임계 알티지 오메가3 비타플러스 3개월 (9,500원/무료) | False | 7      | 23.06.29 19:51:56 | 1         | 0             | 7470 [식품/건강] | https://www.ppomppu.co.kr/zboard/view.php?id=p... | False | False |
| 470669  | 최****    | [옥션] 리큐 진한겔 꿉꿉한냄새 싹 2.1L X 6 [20,930/무료배송]              | False | 23     | 23.06.29 19:39:51 | 1         | 0             | 10084 [기타]    | https://www.ppomppu.co.kr/zboard/view.php?id=p... | False | False |



### 5. 크롤링 코드

```python
# 패키지 import
import requests
import pandas as pd
import bs4
import time
import tqdm

from datetime import datetime
from bs4 import BeautifulSoup

# 뽐뿌 게시판 크롤링 함수 생성
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

# 파라미터 설정
end_page = 5900
base_url = "https://www.ppomppu.co.kr/zboard/zboard.php?id=ppomppu&page={}"
temp_list = []

# 시작 시간 체크
start_time = time.time()

# 크롤링 실행
for page in tqdm.tqdm(range(2, end_page)):
    response = requests.get(base_url.format(page))
    
    # Step1. URL 데이터 요청
    if response.status_code == 200:
        html = BeautifulSoup(response.text, 'html.parser')
        # Step2. HTML 팔싱
        items = html.find_all("tr", ["common-list0", "common-list1"])
        # Step3. 데이터 크롤링
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

# Step4. 데이터 csv 저장
now = str(datetime.now())
df.to_csv(f"./datas/{now}_{len(df)}개.csv", index=False)
```

뽐뿌 게시판을 크롤링하는 코드이며, 위에서 설정한 12개의 데이터를 크롤링할 페이지의 수 만큼 데이터를 가져온다. 특가 게시물이 인기/핫 게시물이 되면 특가 게시판에서 인기/핫 게시판으로 이동되기 때문에 인기/핫 게시판을 한번 더 크롤링해주어 인기/핫 게시물인지 확인하는 과정을 거치었다. 마지막으로 크롤링이 끝나면 데이터를 csv 파일로 저장한다.