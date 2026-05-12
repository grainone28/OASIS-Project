@echo off
echo Starting ERFNet tests on Fishyscapes...

:: 1. MSP with fixed temperatures (PDF Table Page 5)
python evaluate_anomaly.py --method msp --model erfnet --dataset fishyscapes --temperature 0.5
python evaluate_anomaly.py --method msp --model erfnet --dataset fishyscapes --temperature 0.75
python evaluate_anomaly.py --method msp --model erfnet --dataset fishyscapes --temperature 1.1

:: 2. Best T search and image saving (--save-vis)
echo Searching for the best temperatures and saving images...
python evaluate_anomaly.py --method msp --model erfnet --dataset fishyscapes --temperature-search --save-vis
python evaluate_anomaly.py --method maxlogit --model erfnet --dataset fishyscapes --temperature-search --save-vis
python evaluate_anomaly.py --method maxentropy --model erfnet --dataset fishyscapes --temperature-search --save-vis

echo Finished! Check the results in the terminal and the images in the results/ folder