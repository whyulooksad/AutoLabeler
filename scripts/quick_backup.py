#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track-Anything 快速备份工具
提供简单的命令行界面进行项目备份
"""

import os
import sys
import subprocess
from datetime import datetime

def print_banner():
    """打印工具横幅"""
    print("=" * 60)
    print("🚀 Track-Anything 快速备份工具")
    print("=" * 60)

def print_menu():
    """打印菜单选项"""
    print("\n请选择备份类型:")
    print("1. 📦 标准备份 (推荐) - 仅源代码和配置")
    print("2. 📦 完整备份 - 包含所有文件")
    print("3. 📋 查看现有备份")
    print("4. 🗑️  管理备份文件")
    print("5. ❓ 查看帮助")
    print("0. 🚪 退出")
    print("-" * 60)

def run_standard_backup():
    """运行标准备份"""
    print("\n📦 开始标准备份...")
    try:
        result = subprocess.run([sys.executable, "backup_project.py"], 
                              capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print("✅ 标准备份完成!")
            print(result.stdout)
        else:
            print("❌ 标准备份失败!")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 备份过程中出错: {e}")

def run_full_backup():
    """运行完整备份"""
    print("\n📦 开始完整备份...")
    print("⚠️  注意: 完整备份文件较大，可能需要较长时间")
    
    confirm = input("确认进行完整备份? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 取消完整备份")
        return
    
    try:
        result = subprocess.run([sys.executable, "backup_project_full.py", "--full"], 
                              capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print("✅ 完整备份完成!")
            print(result.stdout)
        else:
            print("❌ 完整备份失败!")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 备份过程中出错: {e}")

def list_backups():
    """列出现有备份"""
    print("\n📋 现有备份文件:")
    try:
        result = subprocess.run([sys.executable, "backup_manager.py", "--list"], 
                              capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ 无法获取备份列表")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 获取备份列表时出错: {e}")

def manage_backups():
    """管理备份文件"""
    print("\n🗑️  备份管理选项:")
    print("1. 查看备份详细信息")
    print("2. 删除备份文件")
    print("3. 清理旧备份")
    print("4. 恢复备份")
    print("0. 返回主菜单")
    
    choice = input("\n请选择操作: ").strip()
    
    if choice == "1":
        backup_file = input("请输入备份文件路径: ").strip()
        if backup_file:
            subprocess.run([sys.executable, "backup_manager.py", "--info", backup_file])
    
    elif choice == "2":
        backup_file = input("请输入要删除的备份文件路径: ").strip()
        if backup_file:
            subprocess.run([sys.executable, "backup_manager.py", "--delete", backup_file])
    
    elif choice == "3":
        count = input("保留最新的几个备份文件? (默认5): ").strip()
        if not count:
            count = "5"
        subprocess.run([sys.executable, "backup_manager.py", "--cleanup", count])
    
    elif choice == "4":
        backup_file = input("请输入要恢复的备份文件路径: ").strip()
        if backup_file:
            restore_dir = input("恢复目录 (可选，留空使用默认): ").strip()
            if restore_dir:
                subprocess.run([sys.executable, "backup_manager.py", "--restore", backup_file, "--restore-dir", restore_dir])
            else:
                subprocess.run([sys.executable, "backup_manager.py", "--restore", backup_file])
    
    elif choice == "0":
        return
    
    else:
        print("❌ 无效选择")

def show_help():
    """显示帮助信息"""
    print("\n❓ 帮助信息:")
    print("=" * 50)
    print("📦 标准备份:")
    print("   - 包含源代码、配置文件和核心模块")
    print("   - 文件较小，备份速度快")
    print("   - 适合日常备份和代码保护")
    print()
    print("📦 完整备份:")
    print("   - 包含所有文件，包括模型文件")
    print("   - 文件较大，备份时间长")
    print("   - 适合完整项目迁移")
    print()
    print("📋 备份管理:")
    print("   - 查看备份文件列表和详细信息")
    print("   - 删除不需要的备份文件")
    print("   - 清理旧备份以节省空间")
    print("   - 恢复备份到指定目录")
    print()
    print("💡 使用建议:")
    print("   - 定期进行标准备份")
    print("   - 重要更新后进行完整备份")
    print("   - 定期清理旧备份文件")
    print("   - 重要备份建议复制到外部存储")

def check_dependencies():
    """检查依赖文件是否存在"""
    required_files = ["backup_project.py", "backup_project_full.py", "backup_manager.py"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ 缺少必要的备份工具文件:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n请确保所有备份工具文件都在当前目录中")
        return False
    
    return True

def main():
    """主函数"""
    print_banner()
    
    # 检查依赖文件
    if not check_dependencies():
        return
    
    while True:
        print_menu()
        choice = input("请输入选择 (0-5): ").strip()
        
        if choice == "1":
            run_standard_backup()
        elif choice == "2":
            run_full_backup()
        elif choice == "3":
            list_backups()
        elif choice == "4":
            manage_backups()
        elif choice == "5":
            show_help()
        elif choice == "0":
            print("\n👋 感谢使用 Track-Anything 备份工具!")
            break
        else:
            print("❌ 无效选择，请重新输入")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        print("请检查错误信息并重试")

