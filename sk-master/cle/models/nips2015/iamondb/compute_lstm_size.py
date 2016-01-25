# A script in order to compute the size (w.r.t to the number of parameters) of each models

frame_size = 200
latent_size = 200

main_lstm_dim = 1200
x2s_dim = 250

main_lstm_param = (x2s_dim * main_lstm_dim * 4 +
                   main_lstm_dim**2 * 4 +
                   main_lstm_dim * 4)

print "Single layer parameter size: %d MB" % (main_lstm_param * 4 / (1024**2))


lstm_0_dim = 560
lstm_1_dim = 560
lstm_2_dim = 560

lstm_0_param = (x2s_dim * lstm_0_dim * 4 +
                lstm_0_dim**2 * 4 +
                lstm_0_dim * 4)
lstm_1_param = (lstm_0_dim * lstm_1_dim * 4 +
                lstm_1_dim**2 * 4 +
                lstm_1_dim * 4)
lstm_2_param = (lstm_2_dim * lstm_2_dim * 4 +
                lstm_2_dim**2 * 4 +
                lstm_2_dim * 4)

lstm_param = lstm_0_param + lstm_1_param + lstm_2_param

print "Deep layers parameter size: %d MB" % (lstm_param * 4 / (1024**2))
