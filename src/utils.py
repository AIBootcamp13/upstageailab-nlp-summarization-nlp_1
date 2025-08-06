import os
import logging
from datetime import datetime, timezone, timedelta


class CustomFormatter(logging.Formatter):
    """한국 시간대(UTC+9)로 로그 포맷을 설정하는 커스텀 포매터"""
    def format(self, record):
        # UTC+9 서울 타임존 설정
        kst = timezone(timedelta(hours=9))
        timestamp = datetime.fromtimestamp(record.created, tz=kst).strftime("%y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [INFO] {record.getMessage()}"


def setup_logger(script_file_path=None, file_mode="w"):
    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러가 있다면 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    
    # 파일 핸들러 (스크립트명.log로 저장)
    if script_file_path is None:
        import sys
        # __main__ 모듈의 __file__ 속성을 사용하여 실제 실행중인 스크립트 경로 얻기
        import __main__
        if hasattr(__main__, '__file__'):
            script_file_path = __main__.__file__
        else:
            script_file_path = sys.argv[0] if sys.argv[0] else __file__
    
    # 절대 경로로 변환
    script_file_path = os.path.abspath(script_file_path)
    script_name = os.path.splitext(os.path.basename(script_file_path))[0]
    script_dir = os.path.dirname(script_file_path)
    
    # 로그 파일이 log_util.py의 이름으로 저장되는 것을 방지
    if script_name == 'log_util':
        import __main__
        if hasattr(__main__, '__file__'):
            main_script_path = os.path.abspath(__main__.__file__)
            script_name = os.path.splitext(os.path.basename(main_script_path))[0]
            script_dir = os.path.dirname(main_script_path)
    
    log_file_path = os.path.join(script_dir, f"{script_name}.log")
    
    # 로그 파일의 디렉토리가 없으면 생성
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file_path, mode=file_mode, encoding='utf-8')
    file_handler.setFormatter(CustomFormatter())
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    
    return logger


# 모듈 로드 시 자동으로 로거 초기화 (스크립트 파일 경로 자동 감지, 기본 모드 "w")
_logger = setup_logger()
_current_script_path = None
_current_file_mode = "w"

# 모듈 레벨에서 로그 메서드들을 직접 노출
def info(*args):
    message = ' '.join(str(arg) for arg in args)
    _logger.info(message)

def debug(*args):
    message = ' '.join(str(arg) for arg in args)
    _logger.debug(message)

def warning(*args):
    message = ' '.join(str(arg) for arg in args)
    _logger.warning(message)

def error(*args):
    message = ' '.join(str(arg) for arg in args)
    _logger.error(message)

def critical(*args):
    message = ' '.join(str(arg) for arg in args)
    _logger.critical(message)

# 로거 객체도 직접 노출 (필요시 사용)
logger = _logger


def change_file_mode(file_mode):
    global _logger, logger, _current_script_path, _current_file_mode
    
    _current_file_mode = file_mode
    
    # 현재 스크립트 경로가 설정되어 있으면 그것을 사용, 없으면 기본값 사용
    script_path = _current_script_path if _current_script_path else None
    
    _logger = setup_logger(script_path, file_mode)
    logger = _logger
