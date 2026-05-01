#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目备份脚本
自动备份Track-Anything项目的所有重要文件和目录
"""

import os
import shutil
import zipfile
from datetime import datetime
import logging

def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('backup.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def create_backup():
    """创建项目备份"""
    logger = setup_logging()
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"track_anything_backup_{timestamp}"
    
    # 创建备份目录
    backup_dir = f"./backups/{backup_name}"
    os.makedirs(backup_dir, exist_ok=True)
    
    logger.info(f"开始创建项目备份: {backup_name}")
    logger.info(f"备份目录: {backup_dir}")
    
    # 定义需要备份的文件和目录
    files_to_backup = [
        "app.py",
        "demo.py",
        "check_gpu.py",
        "requirements.txt",
        "README.md",
        "LICENSE",
        "LOGGING_README.md",
        "=1.13.3"  # 版本文件
    ]
    
    dirs_to_backup = [
        "tracker",
        "inpainter", 
        "tools",
        "templates",
        "assets",
        "doc",
        "stub_mmcv"
    ]
    
    # 备份文件
    logger.info("正在备份文件...")
    for file_name in files_to_backup:
        if os.path.exists(file_name):
            try:
                shutil.copy2(file_name, backup_dir)
                logger.info(f"✓ 已备份文件: {file_name}")
            except Exception as e:
                logger.error(f"✗ 备份文件失败 {file_name}: {e}")
        else:
            logger.warning(f"⚠ 文件不存在: {file_name}")
    
    # 备份目录
    logger.info("正在备份目录...")
    for dir_name in dirs_to_backup:
        if os.path.exists(dir_name):
            try:
                dest_dir = os.path.join(backup_dir, dir_name)
                shutil.copytree(dir_name, dest_dir, dirs_exist_ok=True)
                logger.info(f"✓ 已备份目录: {dir_name}")
            except Exception as e:
                logger.error(f"✗ 备份目录失败 {dir_name}: {e}")
        else:
            logger.warning(f"⚠ 目录不存在: {dir_name}")
    
    # 创建备份信息文件
    backup_info = f"""Track-Anything 项目备份信息
=======================

备份时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
备份名称: {backup_name}
原始目录: {os.getcwd()}

备份内容:
- 主要Python文件 (app.py, demo.py, check_gpu.py)
- 配置文件 (requirements.txt, README.md, LICENSE)
- 核心模块目录 (tracker/, inpainter/, tools/)
- 模板和资源文件 (templates/, assets/, doc/)
- 其他重要文件

注意事项:
- 此备份不包含checkpoints/目录（模型文件较大）
- 不包含logs/目录（日志文件）
- 不包含result/目录（结果文件）
- 不包含temp_*/目录（临时文件）
- 不包含track/目录（虚拟环境）

如需完整备份，请手动复制上述目录。
"""
    
    with open(os.path.join(backup_dir, "BACKUP_INFO.txt"), "w", encoding="utf-8") as f:
        f.write(backup_info)
    
    # 创建ZIP压缩包
    zip_path = f"{backup_dir}.zip"
    logger.info(f"正在创建ZIP压缩包: {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(backup_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, backup_dir)
                    zipf.write(file_path, arcname)
        
        logger.info(f"✓ ZIP压缩包创建成功: {zip_path}")
        
        # 删除临时备份目录，只保留ZIP文件
        shutil.rmtree(backup_dir)
        logger.info("✓ 已清理临时备份目录")
        
    except Exception as e:
        logger.error(f"✗ 创建ZIP压缩包失败: {e}")
        logger.info(f"备份文件保存在: {backup_dir}")
    
    # 显示备份统计信息
    logger.info("=" * 50)
    logger.info("备份完成!")
    logger.info(f"备份文件: {zip_path}")
    logger.info(f"备份大小: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")
    logger.info("=" * 50)
    
    return zip_path

if __name__ == "__main__":
    try:
        backup_file = create_backup()
        print(f"\n✅ 项目备份完成!")
        print(f"📁 备份文件: {backup_file}")
        print(f"📊 文件大小: {os.path.getsize(backup_file) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"❌ 备份失败: {e}")
        logging.error(f"备份过程中出现错误: {e}")

