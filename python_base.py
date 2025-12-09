#%% 
"""퀀트 투자 포트폴리오 만들기""" 
'''퀀트 투자 포트폴리오 만들기'''

# %% 따옴표를 3 개 사용할 경우 줄 바꿈을 통해 여러 줄의 문자열을 만들 수 있다.
multiline = """Life is too short 
You need python""" 

print(multiline)

# %% 문자열 맨 앞에 f를 붙이면 f-string 포매팅이 선언되며, 중괄호 안에 
#    변수를 넣으면 해당 부분에 변수의 값이 입력 된다 .
name = '조문수'
birth = '1984' 

f'나의 이름은 {name}이며, {birth}년에 태어났습니다.'

# %% 문자열 길이 구하기: len()
a = 'Life is too short' 
len(a)

# %% 문자열 치환하기: replace()
#    문자열.replace(a, b): a를 b로 치환
var = '퀀트 투자 포트폴리오 만들기' 
var.replace (' ', '_')


# %% 문자열 나누기: split()
var = '퀀트 투자 포트폴리오 만들기' 
var.split(' ')

# %% 문자열 인덱싱(파이썬은 숫자가 0부터 시작 → 2는 세번째 문자)
var = 'Quant'
var[2]

# %% 인덱싱에 마이너스(-) 기호를 붙이면 문자열 뒤에서 부터 읽는다.
var[-2]

# %% 슬라이싱 var[시작:마지막], 이때 마지막 번호는 미포함
#    var[0:3] → 0이상 3미만(첫번째 부터 세번째 까지)
var[:3]

# %%
var[3:]

# %% 리스트: 연속된 데이터 표현, 대괄호([]) 이용해 생성
#    리스트_숫자
list_num = [1, 2, 3]
print(list_num)

# %% 리스트_문자열
list_char = ['a', 'b', 'c']
print(list_char)

# %% 리스트_복합
list_complex = [1, 'a', 2]
print(list_complex)

# %% 리스트_중첩
list_nest = [1, 2, ['a', 'b']]
print(list_nest)

# %% 리스트 인덱싱
var = [1, 2, ['a', 'b', 'c']]
var[0]

# %% 리스트 인덱싱
var[2]

# %% 중첩된 리스트에서 내부 리스트 인덱싱
var[2][0]

# %% 리스트 슬라이싱
var = [1, 2, 3, 4, 5]
var[0:2]

# %% 리스트 연산 + : 두 리스트를 하나로 합치기
a = [1, 2, 3]
b = [4, 5, 6]

a + b

# %% 리스트 연산 * : 해당 리스트를 n번 반복
a = [1, 2, 3]
a * 3

# %% 리스트에 요소 추가 append() : 리스트 마지막에 데이터 추가
var = [1, 2, 3]
var.append(4)
print(var)

# %% append() 함수 내에 리스트 입력 시 중첩된 형태로 추가
var = [1, 2, 3]
var.append([4, 5])
print(var)

# %% extend() : 리스트 내에 중첩된 형태가 아닌 단일 리스트로 확장
var = [1, 2, 3]
var.extend([4, 5])
print(var)

# %% 리스트 수정: 인덱싱 이용
var = [1, 2, 3, 4, 5]
var[2] = 10
print(var)

# %% 리스트 수정: 인덱싱 이용
var[3] = ['a', 'b', 'c']
print(var)

# %% 리스트 수정: 슬라이싱 이용
var[0 : 2] = ['가', '나']
print(var)

# %% 리스트 내 요소 삭제: del 명령어 / 인덱싱
var = [1, 2, 3]
del var[0]
print(var)

# %% 리스트 내 요소 삭제: del 명령어 / 슬라이싱
var = [1, 2, 3]
del var[0:2]
print(var)

# %% 슬라이싱으로 범위 지정 후 빈 리스트([]) 입력 시 데이터 삭제
var = [1, 2, 3]
var[0:2] = []
print(var)

# %% remove(x) : 리스트에서 첫 번째로 나오는 x를 삭제
var = [1, 2, 3]
var.remove(1)
print(var)

# %% pop() : 리스트의 맨 마지막 요소를 반환하고 해당 요소는 삭제
var = [1, 2, 3]
var.pop()
print(var)

# %% sort() : 리스트 내 데이터 정렬(오름차순)
var = [2, 4, 1, 3]
var.sort()
print(var)

# %% 튜플: 리스트와 흡사하나 소괄호(()) 이용해 생성하고, 값 수정 및 삭제 불가
#    값이 하나인 경우 값 뒤에 반드시 콤마(,) 입력
var = (1, )
print(var)

# %% 튜플도 중첩 가능
var = (1, 2, ('a', 'b'))
print(var)

# %% 튜플은 값 삭제 불가
var = (1, 2)
del var[0]

# %% 튜플도 인덱싱이나 슬라이싱은 리스트와 동일
var = (1, 2, 3)
var[0]

# %%  딕셔너리: 대응 관계를 나타내는 자료형, 중괄호({})를 이용해 생성하며 
#     키:값 형태
var = {'key1' : 1, 'key2' : 2}
print(var)

# %% 딕셔너리 값으로 리스트나 튜플 형태도 가능
var = {'key1' : [1, 2, 3], 'key2' : ('a', 'b', 'c')}
print(var)

# %% 딕셔너리 키를 사용해 값 구하기
var = {'key1' : 1, 'key2' : 2, 'key3' : 3}
var['key1']

# %% 딕셔너리 쌍 추가하기(예시: 키는 key3, 값은 3인 딕셔너리 쌍 추가)
var = {'key1': 1, 'key2': 2} 
var['key3'] = 3 
print(var)

# %% 딕셔너리 쌍 삭제: del 명령어
del var['key1'] 
print(var)

# %% 딕셔너리 키와 값을 한번에 구하기
#    var.keys() 입력하면 var라는 딕셔너리에서 키만을 모아 dict_keys 객체로 반환
var = {'key1': 1, 'key2': 2, 'key3': 3}
var.keys()

# %% 결과를 list()로 감싸 주면 키값이 리스트 형태로 출력
list(var.keys())

# %% var.values()는 var라는 딕셔너리 에서 값만을 모아 dict_values 객체 로 반환
var.values()

# %% set() : 집합 자료형
#    set() 내에 리스트를 입력 하면 집합이 만들어진다.
set1 = set([1, 2, 3])
print(set1)

# %% 집합 자료형은 중복 허용 안됨, 순서 없음
set2 = set('banana')
print(set2)

# %%
s1 = set([1, 2, 3, 4])
s2 = set([3, 4, 5, 6])

# %% union() : 합집합
s1.union(s2)

# %% intersection() : 교집합
s1.intersection(s2)

# %% difference() : 차집합
s1.difference(s2)

# %% 불리언 Boolean 자료형: 참 True 또는 거짓 False을 나타내는 자료형
#    True와 False의 첫 문자는 항상 대문자
var = True 
type(var)

# %%
1 == 1

# %%
1 != 2

# %% bool() : True와 False를 반환 하는 함수(숫자 0은 False)
bool(0)

# %%  0이 아닌 숫자는 True
bool(1)

# %%
bool(2)

# %% 날짜, 시간 구하기(datetime 패키지)
import datetime 
var = datetime.datetime.now()
var

# %%
type(var)

# %% 연도
print(var.year)

# %% 월
print(var.month)

# %%
var.year, var.month, var.day, var.hour, var.minute, var.second, var.microsecond

# %% 요일
# 0 : 월요일, 1 : 화요일, 2 : 수요일, 3 : 목요일, 
# 4 : 금요일, 5 : 토요일, 6 : 일요일
var.weekday()

# %% date() : 날짜 관련 정보
var.date()

# %% time() : 시간 관련 정보
var.time()

# %% strftime() : 시간 정보를 문자열 로 바꿈
var.strftime('%Y-%m-%d')

# %%

