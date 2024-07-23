if ( -d results ) then
    rm -rf results
endif

if ( -d saved ) then
    rm -rf saved
endif

python main.py --model LAFTR --fair-coeff 1 --dataset german