import gpflow
import tensorflow as tf
import numpy as np

from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
from sklearn.metrics import f1_score
from gpflow.ci_utils import ci_niter

f64 = gpflow.utilities.to_default_float

def invlink(f): return gpflow.likelihoods.Bernoulli().invlink(f).numpy()

def make_kernel_with_prior(ls):

  lin = gpflow.kernels.Linear()
  lin.variance.prior = tfd.Gamma(f64(2), f64(2))

  rbf = gpflow.kernels.Matern32()
  rbf.variance.prior = tfd.Gamma(f64(2), f64(2))
  rbf.lengthscales.prior = tfd.Gamma(f64(1), f64(ls))

  cons = gpflow.kernels.Constant()
  cons.variance.prior = tfd.Gamma(f64(1), f64(5))

  return  lin * rbf + cons


def GPMC_prediction(m , samples, range_, X):

  Fpred, Ypred = [], []
  for i in range_:
      for var, var_samples in zip(m.trainable_variables, samples):
          var.assign(var_samples[i])
      Ypred.append(m.predict_y(X)[0].numpy().tolist())
      Fpred.append(np.squeeze(invlink(m.predict_f_samples(X, 10))).tolist())

  return (Fpred.mean(1).mean(0) > 0.5).astype(int)



def fit_VGP(data, kern, test_data):

  f1scores = []
  elbos = []

  def train_callback(*args, **kargs):
    pred_train = (invlink(m.predict_f(data[0])[0].numpy().squeeze()) > 0.5).astype(int)
    f1scores.append(f1_score(data[1], pred_train))
    elbos.append(m.elbo().numpy())

  m = gpflow.models.VGP(data,
                        kernel=kern,
                        likelihood=gpflow.likelihoods.Bernoulli())

  opt = gpflow.optimizers.Scipy()
  opt.minimize(m.training_loss, variables=m.trainable_variables,
                      options={"maxiter" : 20},
                      step_callback=train_callback)

  return m, f1scores, elbos


def fit_GPMC(data, kern, test_data):

  burn = 100
  thin = 10

  X_train, y_train = data
  X_test, y_test = test_data

  m = gpflow.models.GPMC(data,
                        kernel=kern,
                        likelihood=gpflow.likelihoods.Bernoulli())

  opt = gpflow.optimizers.Scipy()
  opt.minimize(m.training_loss, variables=m.trainable_variables,
                      options={"maxiter" : 500})


  num_burnin_steps = ci_niter(burn)
  num_samples = ci_niter(400)

  hmc_helper = gpflow.optimizers.SamplingHelper(
      m.log_posterior_density, m.trainable_parameters
  )

  hmc = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
  )

  # adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
  #     hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
  # )

  adaptive_hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
      hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75)
  )

  @tf.function
  def run_chain_fn():
      return tfp.mcmc.sample_chain(
          num_results=num_samples,
          num_burnin_steps=num_burnin_steps,
          current_state=hmc_helper.current_state,
          kernel=adaptive_hmc,
          trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
      )

  samples, _ = run_chain_fn()
  n_samples = len(samples[0])


  obj = []
  for i in range(burn, n_samples, thin):
      for var, var_samples in zip(m.trainable_variables, samples):
          var.assign(var_samples[i])
      obj.append(m.maximum_log_likelihood_objective().numpy())

  f1train = []
  for i in range(burn, n_samples, thin):
      for var, var_samples in zip(m.trainable_variables, samples):
          var.assign(var_samples[i])
      pred_train = (np.squeeze(invlink(m.predict_f_samples(X_train, 10))).mean(0)>0.5).astype(int)
      f1train.append(f1_score(y_train, pred_train))


  pred_test = (invlink(m.predict_f_samples(X_test, 20).numpy().squeeze().T).mean(1) > 0.5).astype(int)


  return m, f1train, obj, f1_score(y_test, pred_test)



































