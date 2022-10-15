#FSHJAR002 MAKEFILE

install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt

venv:
	test -d venv || python3 -m venv venv

clean:
	rm -rf venv
	find -iname "*.pyc" -delete

runMain:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/LSTM_Deep_3" -e 50 -b 128 -m "LSTM_deep3_HighDrop" -n 0.4

runBeat:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/LSTM_Deep_3" -e 50 -b 128 -m "LSTM_deep3_HighDrop" -n 0.4 -x

runLoad:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/LSTM_Deep_3" -e 50 -b 128 -m "LSTM_deep3_HighDrop" -ls

runSkip_recent:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/backup_final_model" -s

runSkip_recent_sim:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/backup_simple_model" -s

runSkipSim:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/simple_model" -s -p oversample normalize

runSkip:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" -s

runRNN:
	venv/bin/python3 src/wrapper.py "data/" "trained_models/simple_model" -ls --m "RNN" -e 10 -p oversample normalize

runTrain:
	venv/bin/python3 src/wrapper.py "data/" "trained_models/simple_model" -b 64 -e 10 -ls

runCNN:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" -e 10 -m "1D_CNN" -ls -p oversample

runLSTM_deep:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" -b 32 -e 5 -m "LSTM_deep" -n 0.4

runLSTM:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" -e 5 -m "LSTM"

runBiLSTM:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" -e 100 -m "BiLSTM"
	
runBiLSTM_pool:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/biLSTM_pool" -ls -e 50 -m "BiLSTM_pool" -p oversample normalize -ls
	
runLSTMSkip:
	venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" --model_choice "LSTM"

runResume:
	venv/bin/python3 src/wrapper.py "data/" "trained_models/final_model" -r

runResume_sim:
	venv/bin/python3 src/wrapper.py "data/" "trained_models/simple_model" -r

runHelp:
	venv/bin/python3 src/wrapper.py -h