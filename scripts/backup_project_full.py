#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track-Anything 项目完整备份脚本
支持标准备份和完整备份两种模式
"""

import os
import shutil
import zipfile
from datetime import datetime
import logging
import argparse

def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('backup_full.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def get_file_size_mb(file_path):
    """获取文件大小（MB）"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except:
        return 0

def get_dir_size_mb(dir_path):
    """获取目录大小（MB）"""
    total_size = 0
    try:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        return total_size / (1024 * 1024)
    except:
        return 0

def create_backup(full_backup=False):
    """创建项目备份"""
    logger = setup_logging()
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_type = "full" if full_backup else "standard"
    backup_name = f"track_anything_{backup_type}_backup_{timestamp}"
    
    # 创建备份目录
    backup_dir = f"./backups/{backup_name}"
    os.makedirs(backup_dir, exist_ok=True)
    
    logger.info(f"开始创建项目备份: {backup_name}")
    logger.info(f"备份类型: {'完整备份' if full_backup else '标准备份'}")
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
        "=1.13.3"
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
    
    # 完整备份时添加的额外目录
    if full_backup:
        extra_dirs = [
            "checkpoints",
            "logs",
            "result",
            "test_sample"
        ]
        dirs_to_backup.extend(extra_dirs)
        logger.info("完整备份模式：将包含模型文件、日志、结果等所有数据")
    
    # 备份文件
    logger.info("正在备份文件...")
    total_files_size = 0
    for file_name in files_to_backup:
        if os.path.exists(file_name):
            try:
                shutil.copy2(file_name, backup_dir)
                file_size = get_file_size_mb(file_name)
                total_files_size += file_size
                logger.info(f"✓ 已备份文件: {file_name} ({file_size:.2f} MB)")
            except Exception as e:
                logger.error(f"✗ 备份文件失败 {file_name}: {e}")
        else:
            logger.warning(f"⚠ 文件不存在: {file_name}")
    
    # 备份目录
    logger.info("正在备份目录...")
    total_dirs_size = 0
    for dir_name in dirs_to_backup:
        if os.path.exists(dir_name):
            try:
                dest_dir = os.path.join(backup_dir, dir_name)
                shutil.copytree(dir_name, dest_dir, dirs_exist_ok=True)
                dir_size = get_dir_size_mb(dir_name)
                total_dirs_size += dir_size
                logger.info(f"✓ 已备份目录: {dir_name} ({dir_size:.2f} MB)")
            except Exception as e:
                logger.error(f"✗ 备份目录失败 {dir_name}: {e}")
        else:
            logger.warning(f"⚠ 目录不存在: {dir_name}")
    
    # 创建备份信息文件
    backup_info = f"""Track-Anything 项目备份信息
=======================

备份时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
备份名称: {backup_name}
备份类型: {'完整备份' if full_backup else '标准备份'}
原始目录: {os.getcwd()}

备份内容:
- 主要Python文件 (app.py, demo.py, check_gpu.py)
- 配置文件 (requirements.txt, README.md, LICENSE)
- 核心模块目录 (tracker/, inpainter/, tools/)
- 模板和资源文件 (templates/, assets/, doc/)
- 其他重要文件

{'完整备份额外包含:' if full_backup else '标准备份不包含:'}
{'  - 模型文件 (checkpoints/)' if full_backup else '  - 模型文件 (checkpoints/) - 文件较大'}
{'  - 日志文件 (logs/)' if full_backup else '  - 日志文件 (logs/) - 运行时生成'}
{'  - 结果文件 (result/)' if full_backup else '  - 结果文件 (result/) - 运行时生成'}
{'  - 测试样本 (test_sample/)' if full_backup else '  - 测试样本 (test_sample/) - 可选'}

备份统计:
- 文件总大小: {total_files_size:.2f} MB
- 目录总大小: {total_dirs_size:.2f} MB
- 预估总大小: {total_files_size + total_dirs_size:.2f} MB

注意事项:
- 此备份包含项目的所有源代码和配置
- 如需恢复，请解压到新目录并安装依赖
- 建议定期备份以保护项目数据
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
    final_size = get_file_size_mb(zip_path)
    logger.info("=" * 60)
    logger.info("备份完成!")
    logger.info(f"备份文件: {zip_path}")
    logger.info(f"备份大小: {final_size:.2f} MB")
    logger.info(f"压缩率: {((total_files_size + total_dirs_size - final_size) / (total_files_size + total_dirs_size) * 100):.1f}%")
    logger.info("=" * 60)
    
    return zip_path, final_size

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Track-Anything 项目备份工具")
    parser.add_argument("--full", action="store_true", help="执行完整备份（包含模型文件等）")
    parser.add_argument("--standard", action="store_true", help="执行标准备份（仅源代码和配置）")
    
    args = parser.parse_args()
    
    # 默认执行标准备份
    full_backup = args.full
    
    if args.standard:
        full_backup = False
    
    print("Track-Anything 项目备份工具")
    print("=" * 40)
    
    if full_backup:
        print("📦 执行完整备份（包含所有文件）")
        print("⚠️  注意：完整备份文件较大，可能需要较长时间")
    else:
        print("📦 执行标准备份（仅源代码和配置）")
        print("💡 提示：如需包含模型文件，请使用 --full 参数")
    
    print("=" * 40)
    
    try:
        backup_file, backup_size = create_backup(full_backup)
        print(f"\n✅ 项目备份完成!")
        print(f"📁 备份文件: {backup_file}")
        print(f"📊 文件大小: {backup_size:.2f} MB")
        print(f"🔧 备份类型: {'完整备份' if full_backup else '标准备份'}")
        
        if backup_size > 100:
            print("💡 提示：备份文件较大，建议使用外部存储设备保存")
        
    except Exception as e:
        print(f"❌ 备份失败: {e}")
        logging.error(f"备份过程中出现错误: {e}")

if __name__ == "__main__":
    main()

