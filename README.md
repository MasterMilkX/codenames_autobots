# AI FOR GAMES CODENAMES COMPETITION BOTS

Experimental bots for the Codenames AI Competition. Baseline bots use term-frequency inverse-document frequency algorithms and the Naive Bayes algorithm. Experiment bot uses Transformer neural networks.

## Runner bash scripts

**human_codemaster.sh** - allow the human user to create cluewords for a guesser bot to solve
```
./human_codemaster.sh [GUESSER_BOT PACKAGE LOCATION] [--optional parameters associated with the bot (i.e. corpus data)]
```

**human_guesser.sh** - allows the human user to guess words on the board based on cluewords given by a codemaster bot
```
./human_codemaster.sh [CODEMASTER_BOT PACKAGE LOCATION] [--optional parameters associated with the bot (i.e. corpus data)]
```

