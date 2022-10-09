import pandas as pd

data = {
  'name': [],
  'dialog': []
}

with open('data/chrono_cross_game_script.txt', 'rt') as script_file:
  for line in script_file.readlines():
    if len(line.strip()) == 0: 
      continue
    elif ':' in line:
      data['name'].append(line.replace(":", "").strip())
      data['dialog'].append("")
    else:
      current_dialog_index = len(data['dialog']) - 1
      current_dialog = data['dialog'][current_dialog_index]
      data['dialog'][current_dialog_index] = current_dialog + " " + line.strip()

df = pd.DataFrame(data)
print(df.head())
print(df.tail())