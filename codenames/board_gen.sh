#!/bin/bash
 
for NUM in `seq 0 $1`
do
	python3 make_board.py > boards/board_${NUM}.txt
done