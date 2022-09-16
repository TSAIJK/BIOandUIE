from pprint import pprint
from paddlenlp import Taskflow

schemabio = ['Time', 'Player', 'Competition', 'Score']
## Schema is crucial. When we use 'Time' instead of 'Date' as the prompt, model works not so well. 
Engsen = "China's Gu Ailing won the gold medal with 188.25 points in the women's freestyle ski platform final at the Beijing Winter Olympics on Feb 8, 2018."
CHNsen = "2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌!"

ie = Taskflow('information_extraction',schema = schemabio, model='uie-m-base',schema_lang="en")#uie-m-base

pprint(ie(Engsen))