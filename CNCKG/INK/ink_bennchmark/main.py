import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import subprocess
import tempfile

# 检查数据文件是否存在
def check_data_files(args):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../IBIDI/datasets'))
    centrality_file = os.path.join(data_dir, args.dataset+'_centrality_nodes.csv')
    
    print(f"check the data directory: {data_dir}")
    print(f"check the centrality files: {centrality_file}")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory does not exist: {data_dir}")
        return False
        
    if not os.path.exists(centrality_file):
        print(f"Error: The central file does not exist: {centrality_file}")
        return False

    return True

def run_ink_pipeline(dataset, depth, method, count, level, float_rpr):
    """
    Run the processing flow of the INK project and return the processed training and testing sets
    """
    print("Begin feature representation...")
    if dataset == 'movies':
        # 使用dbpedia_test3.py的代码
        # from DBPedia.dbpedia_test3 import main as dbpedia_main
        #train_file, test_file = dbpedia_main(dataset, depth, method)
        # cmd = f"python DBPedia/dbpedia_test3.py {dataset} {depth} {method} 1 1 1.0"
        # subprocess.run(cmd, shell=True, check=True)
        ink_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        env = os.environ.copy()
        env['PYTHONPATH'] = ink_dir + os.pathsep + env.get('PYTHONPATH', '')
        cmd = [sys.executable, '-m', 'DBPedia.dbpedia_test3', dataset, str(depth), method, count, level, float_rpr]
        subprocess.run(cmd, check=True, env=env)

        output_dir = '../../IBIDI/datasets'
        train_file = os.path.join(output_dir, f'{dataset}_train.csv')
        test_file = os.path.join(output_dir, f'{dataset}_test.csv')
    else:
        # cmd = f"python node_class2.py {dataset} {depth} {method} {count} {level} {float_rpr}"
        cmd = [sys.executable, '-m', 'node_class2', dataset, str(depth), method, count, level, float_rpr]
        subprocess.run(cmd, shell=True, check=True)
        
        # Get output file path
        output_dir = '../../IBIDI/datasets'
        train_file = os.path.join(output_dir, f'{dataset}_train.csv')
        test_file = os.path.join(output_dir, f'{dataset}_test.csv')
    
    # 读取INK处理后的数据
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    return train_data, test_data

def run_bidi_pipeline(dataset, flag, pre, train_data, test_data):
    """
    Run the processing flow of IBIDI project and use INK to process the data
    """
    # Create a temporary directory to store data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the training and testing sets to a temporary file
        train_file = os.path.join(temp_dir, f'{dataset}_train.csv')
        test_file = os.path.join(temp_dir, f'{dataset}_test.csv')
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../IBIDI/datasets'))
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../IBIDI/experiment_logs'))

        # Build command-line parameters for bidi_multicategory.py
        cmd = [
            'python',
            '../../IBIDI/bidi_multicategory.py',
            '--dataset', dataset,
            '--exp_num', '1',
            '--exp_type', '0',  # 使用BIDI算法
            '--param_a', str(pre),
            '--data_dir', data_dir,  # 使用正确的相对路径
            '--log_dir', log_dir
        ]
        
        # 执行命令
        print("Run IBIDI...")
        subprocess.run(cmd, check=True)
        print("End of IBIDI!")

def main():
        
    parser = argparse.ArgumentParser(description='Run CNCKG')
    
    # 创建参数组
    ink_group = parser.add_argument_group('ink args')
    bidi_group = parser.add_argument_group('IBIDI args')
    
    # INK项目参数
    ink_group.add_argument('--dataset', type=str, required=True, help='dataset name')
    ink_group.add_argument('--depth', type=int, default=1, help='depth args for feature representation')
    ink_group.add_argument('--method', type=str, default='INK', help='Using INK for Representation')

    ink_group.add_argument('--count', type=str, default= True, help='whether enable count in feature representation')
    ink_group.add_argument('--level', type=str, default= True, help='whether level count in feature representation')
    ink_group.add_argument('--float_rpr', type=str, default= True, help='whether enable float format in feature representation')
    # IBIDI项目参数
    bidi_group.add_argument('--flag', type=int, default=10, help='number of base classifier for IBIDI')
    bidi_group.add_argument('--pre', type=float, default=1.0, help='alpha for IBIDI')
    
    args = parser.parse_args()

    # 首先检查必需的文件是否存在
    if not check_data_files(args):
        print("Program termination: Required files do not exist")
        sys.exit(1)
    
    # 运行INK项目并获取处理后的数据
    train_data, test_data = run_ink_pipeline(args.dataset, args.depth, args.method, args.count, args.level, args.float_rpr)
    
    # 运行IBIDI项目，使用INK处理后的数据
    run_bidi_pipeline(args.dataset, args.flag, args.pre, train_data, test_data)
    
    print("所有处理完成!")

if __name__ == "__main__":
    main() 
