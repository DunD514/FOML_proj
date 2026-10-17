import os

for root, dirs, files in os.walk('.'):
    for f in files:
        if f == '__init__.py':
            path = os.path.join(root, f)
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write('"""Module initialization."""\n')

print("All __init__.py files sanitized.")
