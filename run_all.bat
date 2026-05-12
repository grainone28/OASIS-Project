@echo off
echo ========================================================
echo   OASIS PROJECT - COMPLETE EVALUATION AUTOMATION
echo ========================================================

:: ----------------------------------------------------------
:: 1. ERFNET - FISHYSCAPES (Lost and Found / Static)
:: ----------------------------------------------------------
echo [TEST] ERFNet on Fishyscapes...
python evaluate_anomaly.py --method msp --model erfnet --dataset fishyscapes --temperature-search --save-vis
python evaluate_anomaly.py --method maxlogit --model erfnet --dataset fishyscapes --temperature-search --save-vis
python evaluate_anomaly.py --method maxentropy --model erfnet --dataset fishyscapes --temperature-search --save-vis

:: ----------------------------------------------------------
:: 2. ERFNET - SMIYC ROAD ANOMALY
:: ----------------------------------------------------------
echo [TEST] ERFNet on SMIYC Road Anomaly...
python evaluate_anomaly.py --method msp --model erfnet --dataset smiyc_anomaly --temperature-search --save-vis
python evaluate_anomaly.py --method maxlogit --model erfnet --dataset smiyc_anomaly --temperature-search --save-vis
python evaluate_anomaly.py --method maxentropy --model erfnet --dataset smiyc_anomaly --temperature-search --save-vis

:: ----------------------------------------------------------
:: 3. ERFNET - SMIYC ROAD OBSTACLE
:: ----------------------------------------------------------
echo [TEST] ERFNet on SMIYC Road Obstacle...
python evaluate_anomaly.py --method msp --model erfnet --dataset smiyc_obstacle --temperature-search --save-vis
python evaluate_anomaly.py --method maxlogit --model erfnet --dataset smiyc_obstacle --temperature-search --save-vis
python evaluate_anomaly.py --method maxentropy --model erfnet --dataset smiyc_obstacle --temperature-search --save-vis

:: ----------------------------------------------------------
:: 4. EOMT - REJECTED BY ALL (RbA)
:: ----------------------------------------------------------
:: Note: Requires eomt_finetuned_best.pth in checkpoints/
echo [TEST] EoMT (RbA) on all datasets...
python evaluate_anomaly.py --method rba --model eomt --dataset fishyscapes --temperature-search --save-vis
python evaluate_anomaly.py --method rba --model eomt --dataset smiyc_anomaly --temperature-search --save-vis
python evaluate_anomaly.py --method rba --model eomt --dataset smiyc_obstacle --temperature-search --save-vis

:: ----------------------------------------------------------
:: 5. SPECIFIC TEMPERATURE TESTS (PDF Page 5 Requirements)
:: ----------------------------------------------------------
echo [TEST] Running specific MSP temperature scales for the report...
python evaluate_anomaly.py --method msp --model erfnet --dataset fishyscapes --temperature 0.5
python evaluate_anomaly.py --method msp --model erfnet --dataset fishyscapes --temperature 0.75
python evaluate_anomaly.py --method msp --model erfnet --dataset fishyscapes --temperature 1.1

echo ========================================================
echo   ALL EXPERIMENTS COMPLETED! 
echo   - Numerical results are in the terminal above.
echo   - Qualitative maps are in: results/visualizations/
echo ========================================================
pause