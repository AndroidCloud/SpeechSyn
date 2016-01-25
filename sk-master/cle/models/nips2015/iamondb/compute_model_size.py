# A script in order to compute the size (w.r.t to the number of parameters) of each models

frame_size = 3
latent_size = 100
n_layer = 2

# M0
main_lstm_dim = 1200
p_x_dim = 250
x2s_dim = 250

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
main_lstm_dim = 1200
p_x_dim = 250
x2s_dim = 250

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
main_lstm_dim = 1200
q_z_dim = 150
p_z_dim = 150
p_x_dim = 200
x2s_dim = 200
z2s_dim = 150

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
lstm_0_dim = 2000
lstm_1_dim = 2000
lstm_2_dim = 2000
q_z_dim = 1000
p_z_dim = 1000
p_x_dim = 1000
x2s_dim = 1000
z2s_dim = 1000
latent_size = 500

lstm_0_param = ((frame_size + 1) * x2s_dim +
                (x2s_dim + 1) * x2s_dim * (n_layer - 1) +
                (latent_size + 1) * z2s_dim +
                (z2s_dim + 1) * z2s_dim * (n_layer - 1) +
                x2s_dim * lstm_0_dim * 4 +
                z2s_dim * lstm_0_dim * 4 +
                lstm_0_dim**2 * 4 +
                lstm_0_dim * 4)
lstm_1_param = (lstm_0_dim * lstm_1_dim * 4 +
                lstm_1_dim**2 * 4 +
                lstm_1_dim * 4)
lstm_2_param = (lstm_2_dim * lstm_2_dim * 4 +
                lstm_2_dim**2 * 4 +
                lstm_2_dim * 4)

lstm_param = lstm_0_param + lstm_1_param + lstm_2_param

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

M3_param = lstm_param + phi_param + prior_param + theta_param

print "M3 parameter size: %d MB" % (M3_param * 4 / (1024**2))
