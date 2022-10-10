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
      data['dialog'][current_dialog_index] = f"{current_dialog} {line.strip()}"

df = pd.DataFrame(data)

# create list of all characters
all_characters = set(data['name'])

# track all characters with more than 30 lines
characters_with_lines = []
for name in all_characters:
  total_lines = sum(df['name'] == name)
  if total_lines > 50:
    characters_with_lines.append(name)

# filter out any characters that have < 30 lines
df = df[df['name'].isin(characters_with_lines)]


df.to_csv('data/formatted_script.csv', index=False)



