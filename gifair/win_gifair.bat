@echo off

echo Run arg: %1

REM set d_list=german compas adult
set d_list=adult
set c_list=0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0
set g_list=0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.5 3.0 3.5 4.0 4.5 5.0
REM set p_list=main evaluate
set p_list=evaluate

for %%d in (%d_list%) do (
	echo Run dataset: %%d
	for %%c in (%c_list%) do (
		echo Run c: %%c
		for %%p in (%p_list%) do (
			echo Run prog: %%p
			REM REM task: DP & B-yNN
			REM python %%p.py --model LAFTR --fair-coeff 1 --fair-coeff-individual %%c --seed %1 --dataset %%d --task DP --lambda 10
			REM python %%p.py --model LAFTR --fair-coeff %%c --fair-coeff-individual 1 --seed %1 --dataset %%d --task DP --lambda 10
			
			REM REM task: EO & B-yNN
			REM python %%p.py --model LAFTR --fair-coeff 1 --fair-coeff-individual %%c --seed %1 --dataset %%d --task EO --lambda 10
			REM python %%p.py --model LAFTR --fair-coeff %%c --fair-coeff-individual 1 --seed %1 --dataset %%d --task EO --lambda 10
			
			REM task: DP & yNN
			python %%p.py --model LAFTR --fair-coeff 1 --fair-coeff-individual %%c --seed %1 --dataset %%d --task DP
			python %%p.py --model LAFTR --fair-coeff %%c --fair-coeff-individual 1 --seed %1 --dataset %%d --task DP
		)
	)
)

for %%d in (%d_list%) do (
	echo Run dataset: %%d
	for %%g in (%g_list%) do (
		echo Run g: %%g
		for %%p in (%p_list%) do (
			echo Run prog: %%p
			REM python %%p.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 0 --seed %1 --dataset %%d --task DP --gamma %%g --lambda 10
			
			REM python %%p.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 0 --seed %1 --dataset %%d --task EO --gamma %%g --lambda 10
			
			python %%p.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 0 --seed %1 --dataset %%d --task DP --gamma %%g
		)
	)
)


REM for %%k in (2 4 6 8 10) do (
    REM for %%l in (0 0.2 0.4 0.6 0.8 1 1.5 2 2.5 3 4 5 6 8 10 15 20) do (
        REM rem for %%s in (0 1 2 3 4) do (
            REM python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 1 --seed %1 --dataset german --task DP --k %%k --lambda %%l
            REM python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 1 --seed %1 --dataset compas --task DP --k %%k --lambda %%l
            REM python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 1 --seed %1 --dataset adult --task DP --k %%k --lambda %%l
        REM rem )
    REM )
REM )


REM for %%c in (0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0) do (
    REM rem for %%s in (0 1 2 3 4) do (
        REM python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual %%c --seed %1 --dataset german --task DP --lambda 10 --aud-steps 0
        REM python main.py --model LAFTR --fair-coeff %%c --fair-coeff-individual 1 --seed %1 --dataset german --task DP --lambda 10 --aud-steps 0

        REM python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual %%c --seed %1 --dataset german --task DP --lambda 10 --aud-individual-steps 0
        REM python main.py --model LAFTR --fair-coeff %%c --fair-coeff-individual 1 --seed %1 --dataset german --task DP --lambda 10 --aud-individual-steps 0
        
        REM python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual %%c --seed %1 --dataset compas --task DP --lambda 10 --aud-steps 0
        REM python main.py --model LAFTR --fair-coeff %%c --fair-coeff-individual 1 --seed %1 --dataset compas --task DP --lambda 10 --aud-steps 0

        REM python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual %%c --seed %1 --dataset compas --task DP --lambda 10 --aud-individual-steps 0
        REM python main.py --model LAFTR --fair-coeff %%c --fair-coeff-individual 1 --seed %1 --dataset compas --task DP --lambda 10 --aud-individual-steps 0
        
        REM python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual %%c --seed %1 --dataset adult --task DP --lambda 10 --aud-steps 0
        REM python main.py --model LAFTR --fair-coeff %%c --fair-coeff-individual 1 --seed %1 --dataset adult --task DP --lambda 10 --aud-steps 0

        REM python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual %%c --seed %1 --dataset adult --task DP --lambda 10 --aud-individual-steps 0
        REM python main.py --model LAFTR --fair-coeff %%c --fair-coeff-individual 1 --seed %1 --dataset adult --task DP --lambda 10 --aud-individual-steps 0
    REM rem )
REM )


rem case studies, do not run and do not change
if %1==999999 (
    rem for %%c in (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0) do (
    rem     for %%s in (0 1 2 3 4) do (
    rem         python main.py --model LAFTR --fair-coeff 0 --fair-coeff-individual %%c --seed %%s --dataset compas --task DP --epoch 400 --gamma 0
    rem     )
    rem )

    rem rem case study 1.1
    rem python main.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 1 --seed 2 --dataset compas --task DP --epoch 400
    rem python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 1 --seed 1 --dataset compas --task DP --epoch 400

    rem rem case study 1.2
    rem python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 0 --seed 1 --dataset compas --task DP --epoch 400
    rem python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 2 --seed 1 --dataset compas --task DP --epoch 400

    rem rem case study 2.1
    rem python main.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 0.1 --seed 0 --dataset adult --task DP --epoch 100
    rem python main.py --model LAFTR --fair-coeff 20 --fair-coeff-individual 1 --seed 4 --dataset adult --task DP --epoch 400

    rem rem case study 2.2
    rem python main.py --model LAFTR --fair-coeff 0.1 --fair-coeff-individual 0 --seed 0 --dataset adult --task DP --epoch 100
    rem python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 0.1 --seed 4 --dataset adult --task DP --epoch 400
rem =============================================================================
rem Testing for GIFair (dataset: adult task: DP seed: 0 coeff group: 0.1，coeff individual: 0.0 k: 10 gamma: 0)
rem =============================================================================
rem start analyzing
rem 628 5872 1 0 1.0138400267348682
rem 628 8459 1 0 1.0138400267348682
rem 628 9407 1 0 1.0138400267348682
rem 628 13996 1 0 1.0138400267348682
rem 628 14147 1 0 1.0138400267348682
rem 2146 2843 0 1 1.0138400267348682
rem 2843 2146 1 0 1.0138400267348682
rem 2843 14589 1 0 1.0138400267348682
rem 4499 13772 1 0 1.013829946292965
rem 5570 5872 1 0 1.0138400267348682
rem 5570 6513 1 0 1.0138400267348682
rem 5570 8459 1 0 1.0138400267348682
rem 5570 9407 1 0 1.0138400267348682
rem 5570 13996 1 0 1.0138400267348682
rem 5570 14147 1 0 1.0138400267348682
rem 5872 628 0 1 1.0138400267348682
rem 5872 5570 0 1 1.0138400267348682
rem 6513 628 0 1 1.0138400267348682
rem 6513 5570 0 1 1.0138400267348682
rem 7597 13772 1 0 1.0104016552433628
rem 8459 628 0 1 1.0138400267348682
rem 8459 5570 0 1 1.0138400267348682
rem 9407 628 0 1 1.0138400267348682
rem 9407 5570 0 1 1.0138400267348682
rem 13772 4499 0 1 1.013829946292965
rem 13772 7597 0 1 1.0104016552433628
rem 13996 628 0 1 1.0138400267348682
rem 13996 5570 0 1 1.0138400267348682
rem 14147 628 0 1 1.0138400267348682
rem 14147 5570 0 1 1.0138400267348682
rem 14589 2843 0 1 1.0138400267348682
rem num_found: 32
rem =============================================================================
rem Testing for GIFair (dataset: adult task: DP seed: 4 coeff group: 1.0，coeff individual: 0.1 k: 10 gamma: 0)
rem =============================================================================
rem start analyzing
rem 1352 8511 0 1 1.0138400267348682
rem 3207 13749 0 1 1.0148938674493926
rem 8511 1352 1 0 1.0138400267348682
rem 13749 3207 1 0 1.0148938674493926
rem num_found: 4

    rem rem case study 3.1
    rem python main.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 1 --seed 3 --dataset german --task DP --epoch 100
    rem python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 10 --seed 3 --dataset german --task DP --epoch 100

    rem for %%c in (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0) do (
    rem     for %%s in (0 1 2 3 4) do (
    rem         python main.py --model LAFTR --fair-coeff %%c --fair-coeff-individual 0 --seed %%s --dataset german --task DP --epoch 100
    rem     )
    rem )
    rem for %%c in (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0) do (
    rem for %%c in (1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0) do (
    rem     for %%s in (0 1 2 3 4) do (
    rem         python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual %%c --seed %%s --dataset german --task DP --epoch 100
    rem     )
    rem )

    rem python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 0 --seed 0 --dataset german --task DP --epoch 100
    rem python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 0 --seed 1 --dataset german --task DP --epoch 100
    rem python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 0 --seed 2 --dataset german --task DP --epoch 100
    python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 0 --seed 3 --dataset german --task DP --epoch 100
    rem python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 0 --seed 4 --dataset german --task DP --epoch 100

    python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 10 --seed 0 --dataset german --task DP --epoch 100

rem =============================================================================
rem Testing for GIFair (dataset: german task: DP seed: 3 coeff group: 1.0，coeff individual: 0.0 k: 10 gamma: 0)
rem =============================================================================
rem start analyzing
rem 3 277 0 1 2.440719467899778
rem 22 47 0 1 2.144674438149556
rem 22 236 0 1 2.4257865485161685
rem 47 22 1 0 2.144674438149556
rem 61 285 0 1 1.9824927198885425
rem 136 265 1 0 2.459783969000481
rem 236 22 1 0 2.4257865485161685
rem 265 136 0 1 2.459783969000481
rem 266 289 0 1 2.478138716338922
rem 277 3 1 0 2.440719467899778
rem 285 61 1 0 1.9824927198885425
rem 289 266 1 0 2.478138716338922
rem num_found: 12
rem =============================================================================
rem Testing for GIFair (dataset: german task: DP seed: 0 coeff group: 1.0，coeff individual: 10.0 k: 10 gamma: 0)
rem =============================================================================
rem start analyzing
rem 3 277 0 1 2.440719467899778
rem 136 265 1 0 2.459783969000481
rem 265 136 0 1 2.459783969000481
rem 277 3 1 0 2.440719467899778
rem num_found: 4
)