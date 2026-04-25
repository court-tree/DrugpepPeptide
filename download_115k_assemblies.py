# -*- coding: utf-8 -*-
import os
import glob
import time
import urllib.request
import urllib.error
import pandas as pd
from multiprocessing import Pool

# ================= 配置区 =================
# 自动扫描这个文件夹下的所有 csv 文件！
CSV_DIR = r"E:\pep\pdb"               
SAVE_DIR = r"E:\pep\dowload"                       
NUM_WORKERS = 8                                    
MAX_RETRIES = 3                                    
TIMEOUT = 30                                       

# 架构师铁律：只下载装配体 mmCIF！
URL_TEMPLATE = "https://files.rcsb.org/download/{}-assembly1.cif"
# =========================================

def download_with_retry(pdb_id):
    """带重试机制的单文件下载"""
    pdb_id = str(pdb_id).strip().lower()
    if len(pdb_id) != 4:
        return None
        
    save_path = os.path.join(SAVE_DIR, f"{pdb_id}.cif")
    
    # 【断点续传】如果文件已经存在且大小不为 0，直接跳过
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return f"⏭️ 已存在: {pdb_id}"

    url = URL_TEMPLATE.format(pdb_id)
    
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as response, open(save_path, 'wb') as out_file:
                out_file.write(response.read())
            return f"✅ 成功: {pdb_id}"
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return f"❌ 404 不存在: {pdb_id}"
            time.sleep(2) 
        except Exception as e:
            time.sleep(2) 
            
    return f"⚠️ 失败 (重试{MAX_RETRIES}次): {pdb_id}"

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"📂 正在扫描 {CSV_DIR} 目录下的所有 CSV 文件...")
    # 自动获取所有 .csv 结尾的文件路径
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    
    if not csv_files:
        print("❌ 没找到任何 CSV 文件，请检查 E:\\pep\\pdb 目录！")
        return
        
    print(f"🔍 共找到 {len(csv_files)} 个碎片文件，正在自动合并并提取 ID...")
    
    all_pdb_ids = set() # 使用集合 (set) 自动去重
    
    # 遍历读取所有 csv
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            id_column = df.columns[0]
            raw_ids = df[id_column].astype(str).str.strip().str.lower()
            valid_ids = raw_ids[raw_ids.str.len() == 4].unique().tolist()
            all_pdb_ids.update(valid_ids) # 塞进集合里去重
        except Exception as e:
            print(f"⚠️ 读取文件 {os.path.basename(file)} 时出错: {e}")

    pdb_ids = list(all_pdb_ids)
    total = len(pdb_ids)
    
    if total == 0:
        print("❌ 没有提取到任何合法的 PDB ID！")
        return

    print(f"🎯 成功合并！自动去重后共获得 {total} 个合法 PDB ID。")
    print(f"🚀 准备开启 {NUM_WORKERS} 进程火力覆盖...")
    print(f"💾 下载目录: {SAVE_DIR}")
    print("-" * 50)
    
    success_count = 0
    with Pool(NUM_WORKERS) as pool:
        for i, msg in enumerate(pool.imap_unordered(download_with_retry, pdb_ids), 1):
            if msg:
                print(f"[{i}/{total}] {msg}")
                if "✅ 成功" in msg or "⏭️ 已存在" in msg:
                    success_count += 1

    print("-" * 50)
    print(f"🎉 批量下载任务结束！有效入库率: {success_count}/{total}")

if __name__ == "__main__":
    main()