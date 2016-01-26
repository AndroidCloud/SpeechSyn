from audio_tools import run_phase_reconstruction_example

if __name__ == "__main__":
	
	for i in range(70,6063):
		run_phase_reconstruction_example("ground_truth/ground_truth_" + str(i+1) + ".wav", "test_" + str(i+1));
	        if i%100 == 0:
			print i
		
