with open('app/main.py', 'r') as f:
    lines = f.readlines()

for i in range(390, 488):
    if lines[i].strip():
        lines[i] = "    " + lines[i]

with open('app/main.py', 'w') as f:
    f.writelines(lines)
