import logging
import logging.handlers

# Logger 인스턴스 생성 및 로그 레벨 ㅅ설정
logger = logging.getLogger("mylogger")
logger.setLevel(logging.DEBUG)

# formmatter생성
# formatter = logging.Formatter('[%(Levelname)s:%filename)s:%(lineno)s)]%(asctime)s>%(message)s')
formatter = logging.Formatter('[%(filename)s:%(lineno)s)]%(asctime)s>%(message)s')

# filehander 와 StreamHander생성
# fileHander = logging.FileHandler('../log/my.log', 'a', 'utf-8')  #한글깨짐 주의

fileMaxByte = 1024 * 1024 * 100 #100MB
fileHandler = logging.handlers.RotatingFileHandler('./log/my.log', maxBytes=fileMaxByte, backupCount=10, encoding='utf-8')
streamHandler = logging.StreamHandler()

# handler에 formmater 셋팅
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)

# Handler 를 logging에 추가
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)