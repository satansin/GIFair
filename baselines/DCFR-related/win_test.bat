if exist results rd /s /q results
if exist saved rd /s /q saved

python main.py --model LAFTR --fair-coeff 1 --dataset german