
data = {
  'name': [],
  'dialog': []
}

with open('data/chrono_cross_game_script.txt', 'rt') as script_file:
  for line in script_file.readlines():
    print(f"|{line}|")