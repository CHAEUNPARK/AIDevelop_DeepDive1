# import re
#
# text= "에러 1122 : 레퍼런스 오류 \n에러 1033: 아규먼트 오류\n에러 1033: 싸이콘서트"
# regex = re.compile("에러 1033")
# mo = regex.search(text)
# if mo != None:
#     print(mo.group())

import re

text = "문의사항이 있으면 032-232-3245 으로 연락주시기 바랍니다."

regex = re.compile(r'(?P<area>\d{3})-(?P<num>\d{3}-\d{4})')
matchoobj = regex.search(text)
areacode = matchoobj.group("area")
num = matchoobj.group("num")
print(areacode,num)

# lambda -> 함수 자체를 인자로 받는 함수에 쓰인다.라고 정리
# python comprehension
# set은 순서가 정의되어 있지 않다.

