#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track-Anything 备份管理工具
用于查看、删除和管理备份文件
"""

import os
import glob
import zipfile
from datetime import datetime
import argparse
import shutil

def get_backup_files():
    """获取所有备份文件"""
    backup_pattern = "./backups/track_anything_*_backup_*.zip"
    backup_files = glob.glob(backup_pattern)
    return sorted(backup_files, reverse=True)

def get_file_info(file_path):
    """获取文件信息"""
    try:
        stat = os.stat(file_path)
        size_mb = stat.st_size / (1024 * 1024)
        modified_time = datetime.fromtimestamp(stat.st_mtime)
        return {
            'size_mb': size_mb,
            'modified_time': modified_time,
            'exists': True
        }
    except:
        return {
            'size_mb': 0,
            'modified_time': None,
            'exists': False
        }

def list_backups():
    """列出所有备份文件"""
    backup_files = get_backup_files()
    
    if not backup_files:
        print("📭 没有找到备份文件")
        return
    
    print("📦 备份文件列表:")
    print("=" * 80)
    print(f"{'序号':<4} {'文件名':<50} {'大小(MB)':<10} {'修改时间':<20}")
    print("-" * 80)
    
    total_size = 0
    for i, backup_file in enumerate(backup_files, 1):
        file_info = get_file_info(backup_file)
        if file_info['exists']:
            filename = os.path.basename(backup_file)
            size = file_info['size_mb']
            modified = file_info['modified_time'].strftime("%Y-%m-%d %H:%M:%S")
            total_size += size
            
            print(f"{i:<4} {filename:<50} {size:<10.2f} {modified:<20}")
    
    print("-" * 80)
    print(f"总计: {len(backup_files)} 个备份文件, 总大小: {total_size:.2f} MB")
    print("=" * 80)

def show_backup_info(backup_file):
    """显示备份文件详细信息"""
    if not os.path.exists(backup_file):
        print(f"❌ 备份文件不存在: {backup_file}")
        return
    
    file_info = get_file_info(backup_file)
    if not file_info['exists']:
        print(f"❌ 无法读取文件信息: {backup_file}")
        return
    
    print(f"📋 备份文件详细信息:")
    print("=" * 50)
    print(f"文件名: {os.path.basename(backup_file)}")
    print(f"完整路径: {backup_file}")
    print(f"文件大小: {file_info['size_mb']:.2f} MB")
    print(f"修改时间: {file_info['modified_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 尝试读取备份信息
    try:
        with zipfile.ZipFile(backup_file, 'r') as zipf:
            if 'BACKUP_INFO.txt' in zipf.namelist():
                with zipf.open('BACKUP_INFO.txt') as f:
                    backup_info = f.read().decode('utf-8')
                    print("\n📄 备份信息:")
                    print("-" * 30)
                    print(backup_info)
            else:
                print("\n⚠️  未找到备份信息文件")
    except Exception as e:
        print(f"\n❌ 读取备份信息失败: {e}")

def delete_backup(backup_file):
    """删除备份文件"""
    if not os.path.exists(backup_file):
        print(f"❌ 备份文件不存在: {backup_file}")
        return False
    
    try:
        file_info = get_file_info(backup_file)
        size_mb = file_info['size_mb']
        
        print(f"🗑️  准备删除备份文件:")
        print(f"文件名: {os.path.basename(backup_file)}")
        print(f"大小: {size_mb:.2f} MB")
        
        confirm = input("确认删除? (y/N): ").strip().lower()
        if confirm == 'y':
            os.remove(backup_file)
            print(f"✅ 已删除备份文件: {backup_file}")
            return True
        else:
            print("❌ 取消删除操作")
            return False
    except Exception as e:
        print(f"❌ 删除失败: {e}")
        return False

def cleanup_old_backups(keep_count=5):
    """清理旧备份文件，保留最新的几个"""
    backup_files = get_backup_files()
    
    if len(backup_files) <= keep_count:
        print(f"📦 当前只有 {len(backup_files)} 个备份文件，无需清理")
        return
    
    files_to_delete = backup_files[keep_count:]
    total_size = 0
    
    print(f"🧹 准备清理 {len(files_to_delete)} 个旧备份文件:")
    print("=" * 50)
    
    for i, backup_file in enumerate(files_to_delete, 1):
        file_info = get_file_info(backup_file)
        size_mb = file_info['size_mb']
        total_size += size_mb
        print(f"{i}. {os.path.basename(backup_file)} ({size_mb:.2f} MB)")
    
    print(f"总计将释放空间: {total_size:.2f} MB")
    
    confirm = input(f"确认删除这 {len(files_to_delete)} 个旧备份文件? (y/N): ").strip().lower()
    if confirm == 'y':
        deleted_count = 0
        for backup_file in files_to_delete:
            try:
                os.remove(backup_file)
                deleted_count += 1
                print(f"✅ 已删除: {os.path.basename(backup_file)}")
            except Exception as e:
                print(f"❌ 删除失败 {os.path.basename(backup_file)}: {e}")
        
        print(f"🎉 清理完成! 删除了 {deleted_count} 个文件，释放了 {total_size:.2f} MB 空间")
    else:
        print("❌ 取消清理操作")

def restore_backup(backup_file, restore_dir=None):
    """恢复备份文件"""
    if not os.path.exists(backup_file):
        print(f"❌ 备份文件不存在: {backup_file}")
        return False
    
    if restore_dir is None:
        restore_dir = f"./restored_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        print(f"📦 正在恢复备份到: {restore_dir}")
        
        with zipfile.ZipFile(backup_file, 'r') as zipf:
            zipf.extractall(restore_dir)
        
        print(f"✅ 备份恢复完成!")
        print(f"📁 恢复目录: {restore_dir}")
        print(f"💡 提示: 请检查恢复的文件，然后安装依赖: pip install -r requirements.txt")
        
        return True
    except Exception as e:
        print(f"❌ 恢复失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Track-Anything 备份管理工具")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有备份文件")
    parser.add_argument("--info", "-i", type=str, help="显示指定备份文件的详细信息")
    parser.add_argument("--delete", "-d", type=str, help="删除指定的备份文件")
    parser.add_argument("--cleanup", "-c", type=int, metavar="COUNT", help="清理旧备份文件，保留最新的COUNT个")
    parser.add_argument("--restore", "-r", type=str, help="恢复指定的备份文件")
    parser.add_argument("--restore-dir", type=str, help="指定恢复目录（与--restore一起使用）")
    
    args = parser.parse_args()
    
    print("Track-Anything 备份管理工具")
    print("=" * 40)
    
    if args.list:
        list_backups()
    elif args.info:
        show_backup_info(args.info)
    elif args.delete:
        delete_backup(args.delete)
    elif args.cleanup is not None:
        cleanup_old_backups(args.cleanup)
    elif args.restore:
        restore_backup(args.restore, args.restore_dir)
    else:
        # 默认显示备份列表
        list_backups()
        print("\n💡 使用 --help 查看所有可用命令")

if __name__ == "__main__":
    main()

