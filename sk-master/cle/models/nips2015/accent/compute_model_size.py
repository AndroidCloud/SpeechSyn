# A script in order to compute the size (w.r.t to the number of parameters) of each models

frame_size = 200
latent_size = 200
n_layer = 4

# M0
main_lstm_dim = 2000
p_x_dim = 600
x2s_dim = 600

main_lstm_param = ((frame_size + 1) * x2s_dim +
                   (x2s_dim + 1) * x2s_dim * (n_layer - 1) +
                   x2s_dim * main_lstm_dim * 4 +
                   main_lstm_dim**2 * 4 +
                   main_lstm_dim * 4)

theta_param = ((main_lstm_dim + 1) * p_x_dim +
               (p_x_dim + 1) *  p_x_dim * (n_layer - 1) +
               (p_x_dim + 1) * frame_size * 2)

M0_param = main_lstm_param + theta_param

print "M0 parameter size: %d MB" % (M0_param * 4 / (1024**2))


# M1
k = 20
main_lstm_dim = 2000
p_x_dim = 450
x2s_dim = 450

main_lstm_param = ((frame_size + 1) * x2s_dim +
                   (x2s_dim + 1) * x2s_dim * (n_layer - 1) +
                   x2s_dim * main_lstm_dim * 4 +
                   main_lstm_dim**2 * 4 +
                   main_lstm_dim * 4)

theta_param = ((main_lstm_dim + 1) * p_x_dim +
               (p_x_dim + 1) *  p_x_dim * (n_layer - 1) +
               (p_x_dim + 1) * frame_size * k * 2 +
               (p_x_dim + 1) * k)

M1_param = main_lstm_param + theta_param

print "M1 parameter size: %d MB" % (M1_param * 4 / (1024**2))


# M2
main_lstm_dim = 2000
q_z_dim = 400
p_z_dim = 400
p_x_dim = 400
x2s_dim = 400
z2s_dim = 400

main_lstm_param = ((frame_size + 1) * x2s_dim +
                   (x2s_dim + 1) * x2s_dim * (n_layer - 1) +
                   (latent_size + 1)* z2s_dim +
                   (z2s_dim + 1) * z2s_dim * (n_layer - 1) +
                   x2s_dim * main_lstm_dim * 4 +
                   z2s_dim * main_lstm_dim * 4 +
                   main_lstm_dim**2 * 4 +
                   main_lstm_dim * 4)

phi_param = (((x2s_dim + main_lstm_dim) + 1) * q_z_dim +
             (q_z_dim + 1) * q_z_dim * (n_layer - 1) +
             (q_z_dim + 1) * latent_size * 2)

prior_param = ((main_lstm_dim + 1) * p_z_dim +
               (p_z_dim + 1) * p_z_dim * (n_layer - 1) +
               (p_z_dim + 1) * latent_size * 2)

theta_param = (((z2s_dim + main_lstm_dim) + 1) * p_x_dim +
               (p_x_dim + 1) * p_x_dim * (n_layer - 1) +
               (p_x_dim + 1) * frame_size * 2)

M2_param = main_lstm_param + phi_param + prior_param + theta_param

print "M2 parameter size: %d MB" % (M2_param * 4 / (1024**2))


# M3
main_lstm_dim = 2000
q_z_dim = 300
p_z_dim = 300
p_x_dim = 400
x2s_dim = 400
z2s_dim = 300

main_lstm_param = ((frame_size + 1) * x2s_dim +
                   (x2s_dim + 1) * x2s_dim * (n_layer - 1) +
                   (latent_size + 1)* z2s_dim +
                   (z2s_dim + 1) * z2s_dim * (n_layer - 1) +
                   x2s_dim * main_lstm_dim * 4 +
                   z2s_dim * main_lstm_dim * 4 +
                   main_lstm_dim**2 * 4 +
                   main_lstm_dim * 4)

phi_param = (((x2s_dim + main_lstm_dim) + 1) * q_z_dim +
             (q_z_dim + 1) * q_z_dim * (n_layer - 1) +
             (q_z_dim + 1) * latent_size * 2)

prior_param = ((main_lstm_dim + 1) * p_z_dim +
               (p_z_dim + 1) * p_z_dim * (n_layer - 1) +
               (p_z_dim + 1) * latent_size * 2)

theta_param = (((z2s_dim + main_lstm_dim) + 1) * p_x_dim +
               (p_x_dim + 1) * p_x_dim * (n_layer - 1) +
               (p_x_dim + 1) * frame_size * k * 2 +
               (p_x_dim + 1) * k)

M3_param = main_lstm_param + phi_param + prior_param + theta_param

print "M3 parameter size: %d MB" % (M3_param * 4 / (1024**2))


# STORN0
encoder_dim = 1400
decoder_dim = 1400
q_z_dim = 400
p_x_dim = 400
x2s_dim = 400
z2s_dim = 400

encoder_param = ((frame_size + 1) * x2s_dim +
                 (x2s_dim + 1) * x2s_dim * (n_layer - 1) +
                 x2s_dim * encoder_dim * 4 +
                 encoder_dim**2 * 4 +
                 encoder_dim * 4)

decoder_param = ((latent_size + 1)* z2s_dim +
                 (z2s_dim + 1) * z2s_dim * (n_layer - 1) +
                 z2s_dim * decoder_dim * 4 +
                 x2s_dim * decoder_dim * 4 +
                 decoder_dim**2 * 4 +
                 decoder_dim * 4)

phi_param = ((encoder_dim + 1) * q_z_dim +
             (q_z_dim + 1) * q_z_dim * (n_layer - 1) +
             (q_z_dim + 1) * latent_size * 2)

theta_param = ((decoder_dim + 1) * p_x_dim +
               (p_x_dim + 1) * p_x_dim * (n_layer - 1) +
               (p_x_dim + 1) * frame_size * 2)

S0_param = encoder_param + decoder_param + phi_param + theta_param

print "STORN0 parameter size: %d MB" % (S0_param * 4 / (1024**2))


# STORN0_ORIG
encoder_dim = 1600
decoder_dim = 1600

encoder_param = ((frame_size + 1) * encoder_dim * 4 +
                 encoder_dim**2 * 4 +
                 encoder_dim * 4)

decoder_param = ((latent_size + 1) * decoder_dim * 4 +
                 (frame_size + 1) * decoder_dim * 4 +
                 decoder_dim**2 * 4 +
                 decoder_dim * 4)

phi_param = (decoder_dim + 1) * latent_size * 2

theta_param = (decoder_dim + 1) * frame_size * 2

S0O_param = encoder_param + decoder_param + phi_param + theta_param

print "STORN0_ORIG parameter size: %d MB" % (S0O_param * 4 / (1024**2))
