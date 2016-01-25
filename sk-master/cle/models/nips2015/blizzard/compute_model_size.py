# A script in order to compute the size (w.r.t to the number of parameters) of each models

frame_size = 200
latent_size = 200
n_layer = 4

# M0
main_lstm_dim = 4000
p_x_dim = 800
x2s_dim = 800

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
main_lstm_dim = 4000
p_x_dim = 700
x2s_dim = 700

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
main_lstm_dim = 4000
q_z_dim = 500
p_z_dim = 500
p_x_dim = 600
x2s_dim = 600
z2s_dim = 500

main_lstm_param = ((frame_size + 1) * x2s_dim +
                   (x2s_dim + 1) * x2s_dim * (n_layer - 1) +
                   (latent_size + 1)* z2s_dim +
                   (z2s_dim + 1) * z2s_dim * (n_layer - 1) +
                   x2s_dim * main_lstm_dim * 4 +
                   z2s_dim * main_lstm_dim * 4 +
                   main_lstm_dim**2 * 4 +
                   main_lstm_dim * 4)

phi_param = ((x2s_dim + main_lstm_dim + 1) * q_z_dim +
             (q_z_dim + 1) * q_z_dim * (n_layer - 1) +
             (q_z_dim + 1) * latent_size * 2)

prior_param = ((main_lstm_dim + 1) * p_z_dim +
               (p_z_dim + 1) * p_z_dim * (n_layer - 1) +
               (p_z_dim + 1) * latent_size * 2)

theta_param = ((z2s_dim + main_lstm_dim + 1) * p_x_dim +
               (p_x_dim + 1) * p_x_dim * (n_layer - 1) +
               (p_x_dim + 1) * frame_size * 2)

M2_param = main_lstm_param + phi_param + prior_param + theta_param

print "M2 parameter size: %d MB" % (M2_param * 4 / (1024**2))


# M3
main_lstm_dim = 4000
q_z_dim = 500
p_z_dim = 500
p_x_dim = 500
x2s_dim = 500
z2s_dim = 500

main_lstm_param = ((frame_size + 1) * x2s_dim +
                   (x2s_dim + 1) * x2s_dim * (n_layer - 1) +
                   (latent_size + 1)* z2s_dim +
                   (z2s_dim + 1) * z2s_dim * (n_layer - 1) +
                   x2s_dim * main_lstm_dim * 4 +
                   z2s_dim * main_lstm_dim * 4 +
                   main_lstm_dim**2 * 4 +
                   main_lstm_dim * 4)

phi_param = ((x2s_dim + main_lstm_dim + 1) * q_z_dim +
             (q_z_dim + 1) * q_z_dim * (n_layer - 1) +
             (q_z_dim + 1) * latent_size * 2)

prior_param = ((main_lstm_dim + 1) * p_z_dim +
               (p_z_dim + 1) * p_z_dim * (n_layer - 1) +
               (p_z_dim + 1) * latent_size * 2)

theta_param = ((z2s_dim + main_lstm_dim + 1) * p_x_dim +
               (p_x_dim + 1) * p_x_dim * (n_layer - 1) +
               (p_x_dim + 1) * frame_size * k * 2 +
               (p_x_dim + 1) * k)

M3_param = main_lstm_param + phi_param + prior_param + theta_param

print "M3 parameter size: %d MB" % (M3_param * 4 / (1024**2))


# M2P
main_lstm_dim = 4000
q_z_dim = 500
p_x_dim = 600
x2s_dim = 600
z2s_dim = 500

main_lstm_param = ((frame_size + 1) * x2s_dim +
                   (x2s_dim + 1) * x2s_dim * (n_layer - 1) +
                   (latent_size + 1)* z2s_dim +
                   (z2s_dim + 1) * z2s_dim * (n_layer - 1) +
                   x2s_dim * main_lstm_dim * 4 +
                   z2s_dim * main_lstm_dim * 4 +
                   main_lstm_dim**2 * 4 +
                   main_lstm_dim * 4)

phi_param = ((x2s_dim + main_lstm_dim + 1) * q_z_dim +
             (q_z_dim + 1) * q_z_dim * (n_layer - 1) +
             (q_z_dim + 1) * latent_size * 2)

theta_param = ((z2s_dim + main_lstm_dim + 1) * p_x_dim +
               (p_x_dim + 1) * p_x_dim * (n_layer - 1) +
               (p_x_dim + 1) * frame_size * 2)

M2P_param = main_lstm_param + phi_param + theta_param

print "M2P parameter size: %d MB" % (M2P_param * 4 / (1024**2))


# M3P
main_lstm_dim = 4000
q_z_dim = 500
p_x_dim = 500
x2s_dim = 500
z2s_dim = 500

main_lstm_param = ((frame_size + 1) * x2s_dim +
                   (x2s_dim + 1) * x2s_dim * (n_layer - 1) +
                   (latent_size + 1)* z2s_dim +
                   (z2s_dim + 1) * z2s_dim * (n_layer - 1) +
                   x2s_dim * main_lstm_dim * 4 +
                   z2s_dim * main_lstm_dim * 4 +
                   main_lstm_dim**2 * 4 +
                   main_lstm_dim * 4)

phi_param = ((x2s_dim + main_lstm_dim + 1) * q_z_dim +
             (q_z_dim + 1) * q_z_dim * (n_layer - 1) +
             (q_z_dim + 1) * latent_size * 2)

theta_param = ((z2s_dim + main_lstm_dim + 1) * p_x_dim +
               (p_x_dim + 1) * p_x_dim * (n_layer - 1) +
               (p_x_dim + 1) * frame_size * k * 2 +
               (p_x_dim + 1) * k)

M3P_param = main_lstm_param + phi_param + theta_param

print "M3P parameter size: %d MB" % (M3P_param * 4 / (1024**2))


frame_size = 200
latent_size = 200


# Deep M0
lstm_0_dim = 3000
lstm_1_dim = 3000
lstm_2_dim = 3000
p_x_dim = 3200
x2s_dim = 3200

lstm_0_param = (x2s_dim * lstm_0_dim * 4 +
                lstm_0_dim**2 * 4 +
                lstm_0_dim * 4)
lstm_1_param = (lstm_0_dim * lstm_1_dim * 4 +
                lstm_1_dim**2 * 4 +
                lstm_1_dim * 4)
lstm_2_param = (lstm_2_dim * lstm_2_dim * 4 +
                lstm_2_dim**2 * 4 +
                lstm_2_dim * 4)

x2s_param = ((frame_size + 1) * x2s_dim +
             (x2s_dim + 1) * x2s_dim * (n_layer - 1))

theta_param = ((main_lstm_dim + 1) * p_x_dim +
               (p_x_dim + 1) * p_x_dim * (n_layer - 1) +
               (p_x_dim + 1) * frame_size * 2)

deep_m0_param = lstm_0_param + lstm_1_param + lstm_2_param + x2s_param + theta_param

print "Deep M0 parameter size: %d MB" % (deep_m0_param * 4 / (1024**2))


# Deep M1
lstm_0_dim = 3000
lstm_1_dim = 3000
lstm_2_dim = 3000
k = 20
p_x_dim = 3000
x2s_dim = 3000

lstm_0_param = (x2s_dim * lstm_0_dim * 4 +
                lstm_0_dim**2 * 4 +
                lstm_0_dim * 4)
lstm_1_param = (lstm_0_dim * lstm_1_dim * 4 +
                lstm_1_dim**2 * 4 +
                lstm_1_dim * 4)
lstm_2_param = (lstm_2_dim * lstm_2_dim * 4 +
                lstm_2_dim**2 * 4 +
                lstm_2_dim * 4)

x2s_param = ((frame_size + 1) * x2s_dim +
             (x2s_dim + 1) * x2s_dim * (n_layer - 1))

theta_param = ((main_lstm_dim + 1) * p_x_dim +
               (p_x_dim + 1) * p_x_dim * (n_layer - 1) +
               (p_x_dim + 1) * frame_size * k * 2 +
               (p_x_dim + 1) * k)

deep_m1_param = lstm_0_param + lstm_1_param + lstm_2_param + x2s_param + theta_param

print "Deep M1 parameter size: %d MB" % (deep_m1_param * 4 / (1024**2))


# Deep M2
lstm_0_dim = 3000
lstm_1_dim = 3000
lstm_2_dim = 3000
q_z_dim = 1500
p_z_dim = 1500
p_x_dim = 2000
x2s_dim = 2000
z2s_dim = 1500

lstm_0_param = (x2s_dim * lstm_0_dim * 4 +
                z2s_dim * lstm_0_dim * 4 +
                lstm_0_dim**2 * 4 +
                lstm_0_dim * 4)
lstm_1_param = (lstm_0_dim * lstm_1_dim * 4 +
                lstm_1_dim**2 * 4 +
                lstm_1_dim * 4)
lstm_2_param = (lstm_2_dim * lstm_2_dim * 4 +
                lstm_2_dim**2 * 4 +
                lstm_2_dim * 4)

x2s_param = ((frame_size + 1) * x2s_dim +
             (x2s_dim + 1) * x2s_dim * (n_layer - 1))

z2s_param = ((latent_size + 1)* z2s_dim +
             (z2s_dim + 1) * z2s_dim * (n_layer - 1))

phi_param = ((x2s_dim + main_lstm_dim + 1) * q_z_dim +
             (q_z_dim + 1) * q_z_dim * (n_layer - 1) +
             (q_z_dim + 1) * latent_size * 2)

prior_param = ((main_lstm_dim + 1) * p_z_dim +
               (p_z_dim + 1) * p_z_dim * (n_layer - 1) +
               (p_z_dim + 1) * latent_size * 2)

theta_param = ((z2s_dim + main_lstm_dim + 1) * p_x_dim +
               (p_x_dim + 1) * p_x_dim * (n_layer - 1) +
               (p_x_dim + 1) * frame_size * 2)

deep_m2_param = lstm_0_param + lstm_1_param + lstm_2_param + x2s_param + z2s_param + phi_param + prior_param + theta_param

print "Deep M2 parameter size: %d MB" % (deep_m2_param * 4 / (1024**2))


# Large M2
main_lstm_dim = 5800
q_z_dim = 1500
p_z_dim = 1500
p_x_dim = 2000
x2s_dim = 2000
z2s_dim = 1500

main_lstm_param = ((frame_size + 1) * x2s_dim +
                   (x2s_dim + 1) * x2s_dim * (n_layer - 1) +
                   (latent_size + 1)* z2s_dim +
                   (z2s_dim + 1) * z2s_dim * (n_layer - 1) +
                   x2s_dim * main_lstm_dim * 4 +
                   z2s_dim * main_lstm_dim * 4 +
                   main_lstm_dim**2 * 4 +
                   main_lstm_dim * 4)

phi_param = ((x2s_dim + main_lstm_dim + 1) * q_z_dim +
             (q_z_dim + 1) * q_z_dim * (n_layer - 1) +
             (q_z_dim + 1) * latent_size * 2)

prior_param = ((main_lstm_dim + 1) * p_z_dim +
               (p_z_dim + 1) * p_z_dim * (n_layer - 1) +
               (p_z_dim + 1) * latent_size * 2)

theta_param = ((z2s_dim + main_lstm_dim + 1) * p_x_dim +
               (p_x_dim + 1) * p_x_dim * (n_layer - 1) +
               (p_x_dim + 1) * frame_size * 2)

large_m2_param = main_lstm_param + phi_param + prior_param + theta_param

print "Large M2 parameter size: %d MB" % (large_m2_param * 4 / (1024**2))


# STORN0
encoder_dim = 2800
decoder_dim = 2800
q_z_dim = 500
p_x_dim = 500
x2s_dim = 600
z2s_dim = 600

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
encoder_dim = 3100
decoder_dim = 3100

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
