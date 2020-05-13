# AI FOR GAMES CODENAMES COMPETITION BOTS

Experimental bots for the Codenames AI Competition. Baseline bots use term-frequency inverse-document frequency algorithms and the Naive Bayes algorithm. Experiment bot uses Transformer neural networks.

## Runner bash scripts

**human_codemaster.sh** - allow the human user to create cluewords for a guesser bot to solve
```
./human_codemaster.sh [GUESSER_BOT PACKAGE LOCATION] [--optional parameters associated with the bot (i.e. corpus data)]
```

**human_guesser.sh** - allows the human user to guess words on the board based on cluewords given by a codemaster bot
```
./human_guesser.sh [CODEMASTER_BOT PACKAGE LOCATION] [--optional parameters associated with the bot (i.e. corpus data)]
```

## Experiment Scripts

**experiment.py** - runs the Transformer, TF-IDF, and NaiveBayes round robin tests using premade boards in the boards/ folder
```
python3 experiment.py
```

**w2vglove_experiment_(codemaster/guesser).py** - runs the W2V+GloVe concatenation bots from the original paper
```
#run the W2V+GloVe codemaster with the Transformer, TF-IDF, and Naive Bayes guesser bots
python3 w2vglove_experiment_codemaster.py 
```
```
#run the W2V+GloVe guesser with the Transformer, TF-IDF, and Naive Bayes codemaster bots
python3 w2vglove_experiment_guesser.py 
```

**gen_boards.sh** - Creates N number of premade game boards in the boards/ folder
```
./gen_boards [NUMBER OF BOARDS TO CREATE]
```

**analyze_bot_results.py** - Analyzes the results from the game generated bot_results.txt file to create .csv files and heatmaps
```
python3 analyze_bot_results.py
```

