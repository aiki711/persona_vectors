import os

def print_tree(startpath, ignore_dirs={'.git', '.venv', 'venv', '__pycache__', '.idea', '.vscode', 'node_modules'}):
    for root, dirs, files in os.walk(startpath):
        # 無視リストにあるディレクトリを除外
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
        print(f'{indent}{os.path.basename(root)}/')
        
        subindent = '│   ' * level + '├── '
        for f in files:
            if not f.startswith('.'): # 隠しファイル（.DS_Storeなど）は除外
                print(f'{subindent}{f}')

if __name__ == "__main__":
    print_tree('.')