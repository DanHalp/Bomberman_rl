import json

import matplotlib.pyplot as plt
import main
import settings
import agents

# First test: time to collect coins
def reset_dict():
    with open("q_table.json", "w") as output:
        output.write(json.dumps(dict()))
        output.close()


games_numbers = [50, 500,1000,2000,3000,4000,5000]
reset_dict()
for num in games_numbers:
    main.s["hui"] = False
    main.s["n_rounds"] = num + 1
    main.game_players = main.set_players(False)[:1]
    main.run_the_game()
    print(agents.steps / num)
    main.game_players = main.set_players(True)[:1]
    main.run_the_game()
    print(agents.steps / num)



settings.training_mode = True
main.game_players = main.set_players()[:3]
print(main.game_players)
x = 3

