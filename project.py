import os
import logging
import sys
from datetime import datetime
from pathlib import Path

# 项目路径配置
project_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_dir, 'model')
data_dir = os.path.join(project_dir, 'data')
output_dir = os.path.join(project_dir, 'output')
output_test_dir = os.path.join(output_dir, 'test')

class ProjectLogger:
    """项目统一的日志管理器"""

    def __init__(self, name=None, log_dir=None, level=logging.INFO):
        self.name = name or 'project'
        self.log_dir = log_dir or os.path.join(output_dir, 'logs')
        self.level = level

        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)

        # 设置日志文件名（包含时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'{self.name}_{timestamp}.log')

        # 初始化日志器
        self.logger = self._setup_logger()

        # 保存原始print函数
        self._original_print = print

        # 重定向print到日志
        self._redirect_print()

    def _setup_logger(self):
        """设置日志器配置"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)

        # 清除已有的处理器
        logger.handlers.clear()

        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(self.level)

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _redirect_print(self):
        """重定向print函数到日志"""
        def log_print(*args, **kwargs):
            # 将print的内容转换为字符串
            message = ' '.join(str(arg) for arg in args)
            self.logger.info(message)
            # 同时输出到控制台
            self._original_print(*args, **kwargs)

        # 替换全局print函数
        import builtins
        builtins.print = log_print

    def info(self, message):
        """记录信息日志"""
        self.logger.info(message)

    def warning(self, message):
        """记录警告日志"""
        self.logger.warning(message)

    def error(self, message):
        """记录错误日志"""
        self.logger.error(message)

    def debug(self, message):
        """记录调试日志"""
        self.logger.debug(message)

    def critical(self, message):
        """记录严重错误日志"""
        self.logger.critical(message)

    def get_log_file(self):
        """获取日志文件路径"""
        return self.log_file

    def close(self):
        """关闭日志器"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

# 全局日志器实例
_global_logger = None

def get_logger(name=None, log_dir=None, level=logging.INFO):
    """获取全局日志器实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ProjectLogger(name, log_dir, level)
    return _global_logger

def setup_logging(name=None, log_dir=None, level=logging.INFO):
    """设置项目日志系统"""
    global _global_logger
    if _global_logger is not None:
        _global_logger.close()
    _global_logger = ProjectLogger(name, log_dir, level)
    return _global_logger

def log_experiment_start(experiment_name, parameters=None):
    """记录实验开始"""
    logger = get_logger()
    logger.info("="*60)
    logger.info(f"🚀 实验开始: {experiment_name}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if parameters:
        logger.info("实验参数:")
        for key, value in parameters.items():
            logger.info(f"  {key}: {value}")
    logger.info("="*60)

def log_experiment_end(experiment_name, duration=None, results=None):
    """记录实验结束"""
    logger = get_logger()
    logger.info("="*60)
    logger.info(f"✅ 实验完成: {experiment_name}")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if duration:
        logger.info(f"总耗时: {duration:.2f}秒")
    if results:
        logger.info("实验结果:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")
    logger.info("="*60)

def log_progress(current, total, message=""):
    """记录进度信息"""
    logger = get_logger()
    percentage = (current / total) * 100 if total > 0 else 0
    logger.info(f"进度: {current}/{total} ({percentage:.1f}%) - {message}")

def log_error(error, context=""):
    """记录错误信息"""
    logger = get_logger()
    logger.error(f"❌ 错误 {context}: {str(error)}")

def log_success(message):
    """记录成功信息"""
    logger = get_logger()
    logger.info(f"✅ {message}")

def log_warning(message):
    """记录警告信息"""
    logger = get_logger()
    logger.warning(f"⚠️ {message}")

# 便捷函数
def info(message):
    """记录信息"""
    get_logger().info(message)

def warning(message):
    """记录警告"""
    get_logger().warning(message)

def error(message):
    """记录错误"""
    get_logger().error(message)

def debug(message):
    """记录调试信息"""
    get_logger().debug(message)

def success(message):
    """记录成功信息"""
    log_success(message)

def progress(current, total, message=""):
    """记录进度"""
    log_progress(current, total, message)
